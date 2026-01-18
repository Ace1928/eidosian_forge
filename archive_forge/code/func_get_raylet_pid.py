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
def get_raylet_pid():
    if sys.platform in ['win32', 'cygwin']:
        return None
    raylet_pid = int(os.environ['RAY_RAYLET_PID'])
    assert raylet_pid > 0
    logger.info('raylet pid is %s', raylet_pid)
    return raylet_pid