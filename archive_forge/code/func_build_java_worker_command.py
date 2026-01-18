import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def build_java_worker_command(bootstrap_address: str, plasma_store_name: str, raylet_name: str, redis_password: str, session_dir: str, node_ip_address: str, setup_worker_path: str):
    """This method assembles the command used to start a Java worker.

    Args:
        bootstrap_address: Bootstrap address of ray cluster.
        plasma_store_name: The name of the plasma store socket to connect
           to.
        raylet_name: The name of the raylet socket to create.
        redis_password: The password of connect to redis.
        session_dir: The path of this session.
        node_ip_address: The ip address for this node.
        setup_worker_path: The path of the Python file that will set up
            the environment for the worker process.
    Returns:
        The command string for starting Java worker.
    """
    pairs = []
    if bootstrap_address is not None:
        pairs.append(('ray.address', bootstrap_address))
    pairs.append(('ray.raylet.node-manager-port', 'RAY_NODE_MANAGER_PORT_PLACEHOLDER'))
    if plasma_store_name is not None:
        pairs.append(('ray.object-store.socket-name', plasma_store_name))
    if raylet_name is not None:
        pairs.append(('ray.raylet.socket-name', raylet_name))
    if redis_password is not None:
        pairs.append(('ray.redis.password', redis_password))
    if node_ip_address is not None:
        pairs.append(('ray.node-ip', node_ip_address))
    pairs.append(('ray.home', RAY_HOME))
    pairs.append(('ray.logging.dir', os.path.join(session_dir, 'logs')))
    pairs.append(('ray.session-dir', session_dir))
    command = [sys.executable] + [setup_worker_path] + ['-D{}={}'.format(*pair) for pair in pairs]
    command += ['RAY_WORKER_DYNAMIC_OPTION_PLACEHOLDER']
    command += ['io.ray.runtime.runner.worker.DefaultWorker']
    return command