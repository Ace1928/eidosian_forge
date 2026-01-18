import asyncio
import io
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import ray
from ray.dashboard.consts import _PARENT_DEATH_THREASHOLD
import ray.dashboard.consts as dashboard_consts
import ray._private.ray_constants as ray_constants
from ray._private.utils import run_background_task
import psutil
def report_raylet_error_logs(log_dir: str, gcs_address: str):
    log_path = os.path.join(log_dir, 'raylet.out')
    error = False
    msg = 'Raylet is terminated. '
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(0, io.SEEK_END)
            pos = max(0, f.tell() - _RAYLET_LOG_MAX_TAIL_SIZE)
            f.seek(pos, io.SEEK_SET)
            raylet_logs = f.readlines()
            if any(('Raylet received SIGTERM' in line for line in raylet_logs)):
                msg += 'Termination is graceful.'
                logger.info(msg)
            else:
                msg += f'Termination is unexpected. Possible reasons include: (1) SIGKILL by the user or system OOM killer, (2) Invalid memory access from Raylet causing SIGSEGV or SIGBUS, (3) Other termination signals. Last {_RAYLET_LOG_MAX_PUBLISH_LINES} lines of the Raylet logs:\n'
                msg += '    ' + '    '.join(raylet_logs[-_RAYLET_LOG_MAX_PUBLISH_LINES:])
                error = True
    except Exception as e:
        msg += f'Failed to read Raylet logs at {log_path}: {e}!'
        logger.exception(msg)
        error = True
    if error:
        logger.error(msg)
        ray._private.utils.publish_error_to_driver(ray_constants.RAYLET_DIED_ERROR, msg, gcs_publisher=ray._raylet.GcsPublisher(address=gcs_address))
    else:
        logger.info(msg)