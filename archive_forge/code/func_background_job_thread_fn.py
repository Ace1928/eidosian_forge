import copy
import yaml
import json
import os
import socket
import sys
import time
import threading
import logging
import uuid
import warnings
import requests
from packaging.version import Version
from typing import Optional, Dict, Tuple, Type
import ray
import ray._private.services
from ray.autoscaler._private.spark.node_provider import HEAD_NODE_ID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray._private.storage import _load_class
from .utils import (
from .start_hook_base import RayOnSparkStartHook
from .databricks_hook import DefaultDatabricksRayOnSparkStartHook
def background_job_thread_fn():
    try:
        _start_ray_worker_nodes(spark=spark, spark_job_group_id=spark_job_group_id, spark_job_group_desc=f'This job group is for spark job which runs the Ray cluster with ray head node {ray_head_ip}:{ray_head_port}', num_worker_nodes=num_worker_nodes, using_stage_scheduling=using_stage_scheduling, ray_head_ip=ray_head_ip, ray_head_port=ray_head_port, ray_temp_dir=ray_temp_dir, num_cpus_per_node=num_cpus_worker_node, num_gpus_per_node=num_gpus_worker_node, heap_memory_per_node=heap_memory_worker_node, object_store_memory_per_node=object_store_memory_worker_node, worker_node_options=worker_node_options, collect_log_to_path=collect_log_to_path, autoscale_mode=False, spark_job_server_port=spark_job_server_port)
    except Exception as e:
        if not ray_cluster_handler.spark_job_is_canceled:
            ray_cluster_handler.background_job_exception = e
            ray_cluster_handler.shutdown(cancel_background_job=False)