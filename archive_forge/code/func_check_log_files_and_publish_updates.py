import argparse
import errno
import glob
import logging
import logging.handlers
import os
import platform
import re
import shutil
import time
import traceback
from typing import Callable, List, Optional, Set
from ray._raylet import GcsClient
import ray._private.ray_constants as ray_constants
import ray._private.services as services
import ray._private.utils
from ray._private.ray_logging import setup_component_logger
def check_log_files_and_publish_updates(self):
    """Gets updates to the log files and publishes them.

        Returns:
            True if anything was published and false otherwise.
        """
    anything_published = False
    lines_to_publish = []

    def flush():
        nonlocal lines_to_publish
        nonlocal anything_published
        if len(lines_to_publish) > 0:
            data = {'ip': self.ip, 'pid': file_info.worker_pid, 'job': file_info.job_id, 'is_err': file_info.is_err_file, 'lines': lines_to_publish, 'actor_name': file_info.actor_name, 'task_name': file_info.task_name}
            try:
                self.publisher.publish_logs(data)
            except Exception:
                logger.exception(f'Failed to publish log messages {data}')
            anything_published = True
            lines_to_publish = []
    for file_info in self.open_file_infos:
        assert not file_info.file_handle.closed
        file_info.reopen_if_necessary()
        max_num_lines_to_read = ray_constants.LOG_MONITOR_NUM_LINES_TO_READ
        for _ in range(max_num_lines_to_read):
            try:
                next_line = file_info.file_handle.readline()
                next_line = next_line.decode('utf-8', 'replace')
                if next_line == '':
                    break
                next_line = next_line.rstrip('\r\n')
                if next_line.startswith(ray_constants.LOG_PREFIX_ACTOR_NAME):
                    flush()
                    file_info.actor_name = next_line.split(ray_constants.LOG_PREFIX_ACTOR_NAME, 1)[1]
                    file_info.task_name = None
                elif next_line.startswith(ray_constants.LOG_PREFIX_TASK_NAME):
                    flush()
                    file_info.task_name = next_line.split(ray_constants.LOG_PREFIX_TASK_NAME, 1)[1]
                elif next_line.startswith(ray_constants.LOG_PREFIX_JOB_ID):
                    file_info.job_id = next_line.split(ray_constants.LOG_PREFIX_JOB_ID, 1)[1]
                elif next_line.startswith(ray_constants.LOG_PREFIX_TASK_ATTEMPT_START) or next_line.startswith(ray_constants.LOG_PREFIX_TASK_ATTEMPT_END):
                    pass
                elif next_line.startswith('Windows fatal exception: access violation'):
                    file_info.file_handle.readline()
                else:
                    lines_to_publish.append(next_line)
            except Exception:
                logger.error(f'Error: Reading file: {file_info.filename}, position: {file_info.file_info.file_handle.tell()} failed.')
                raise
        if file_info.file_position == 0:
            filename = file_info.filename.replace('\\', '/')
            if '/raylet' in filename:
                file_info.worker_pid = 'raylet'
            elif '/gcs_server' in filename:
                file_info.worker_pid = 'gcs_server'
            elif '/monitor' in filename or 'event_AUTOSCALER' in filename:
                file_info.worker_pid = 'autoscaler'
            elif '/runtime_env' in filename:
                file_info.worker_pid = 'runtime_env'
        file_info.file_position = file_info.file_handle.tell()
        flush()
    return anything_published