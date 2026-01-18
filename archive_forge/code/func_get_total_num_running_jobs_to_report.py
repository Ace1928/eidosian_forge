import json
import logging
import threading
import os
import platform
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests
import yaml
import ray
import ray._private.ray_constants as ray_constants
import ray._private.usage.usage_constants as usage_constant
from ray.experimental.internal_kv import _internal_kv_initialized, _internal_kv_put
from ray.core.generated import usage_pb2, gcs_pb2
def get_total_num_running_jobs_to_report(gcs_client) -> Optional[int]:
    """Return the total number of running jobs in the cluster excluding internal ones"""
    try:
        result = gcs_client.get_all_job_info()
        total_num_running_jobs = 0
        for job_info in result.values():
            if not job_info.is_dead and (not job_info.config.ray_namespace.startswith('_ray_internal')):
                total_num_running_jobs += 1
        return total_num_running_jobs
    except Exception as e:
        logger.info(f'Faile to query number of running jobs in the cluster: {e}')
        return None