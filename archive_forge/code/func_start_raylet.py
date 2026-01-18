import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def start_raylet(self, plasma_directory: str, object_store_memory: int, use_valgrind: bool=False, use_profiler: bool=False):
    """Start the raylet.

        Args:
            use_valgrind: True if we should start the process in
                valgrind.
            use_profiler: True if we should start the process in the
                valgrind profiler.
        """
    stdout_file, stderr_file = self.get_log_file_handles('raylet', unique=True)
    process_info = ray._private.services.start_raylet(self.redis_address, self.gcs_address, self._node_ip_address, self._ray_params.node_manager_port, self._raylet_socket_name, self._plasma_store_socket_name, self.cluster_id, self._ray_params.worker_path, self._ray_params.setup_worker_path, self._ray_params.storage, self._temp_dir, self._session_dir, self._runtime_env_dir, self._logs_dir, self.get_resource_spec(), plasma_directory, object_store_memory, self.session_name, is_head_node=self.is_head(), min_worker_port=self._ray_params.min_worker_port, max_worker_port=self._ray_params.max_worker_port, worker_port_list=self._ray_params.worker_port_list, object_manager_port=self._ray_params.object_manager_port, redis_password=self._ray_params.redis_password, metrics_agent_port=self._ray_params.metrics_agent_port, runtime_env_agent_port=self._ray_params.runtime_env_agent_port, metrics_export_port=self._metrics_export_port, dashboard_agent_listen_port=self._ray_params.dashboard_agent_listen_port, use_valgrind=use_valgrind, use_profiler=use_profiler, stdout_file=stdout_file, stderr_file=stderr_file, config=self._config, huge_pages=self._ray_params.huge_pages, fate_share=self.kernel_fate_share, socket_to_use=None, max_bytes=self.max_bytes, backup_count=self.backup_count, ray_debugger_external=self._ray_params.ray_debugger_external, env_updates=self._ray_params.env_vars, node_name=self._ray_params.node_name, webui=self._webui_url, labels=self._get_node_labels())
    assert ray_constants.PROCESS_TYPE_RAYLET not in self.all_processes
    self.all_processes[ray_constants.PROCESS_TYPE_RAYLET] = [process_info]