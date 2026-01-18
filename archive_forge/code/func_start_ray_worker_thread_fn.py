import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pyspark.util import inheritable_thread_target
from ray.util.spark.cluster_init import _start_ray_worker_nodes
def start_ray_worker_thread_fn():
    try:
        _start_ray_worker_nodes(spark=self.server.spark, spark_job_group_id=spark_job_group_id, spark_job_group_desc=spark_job_group_desc, num_worker_nodes=1, using_stage_scheduling=using_stage_scheduling, ray_head_ip=ray_head_ip, ray_head_port=ray_head_port, ray_temp_dir=ray_temp_dir, num_cpus_per_node=num_cpus_per_node, num_gpus_per_node=num_gpus_per_node, heap_memory_per_node=heap_memory_per_node, object_store_memory_per_node=object_store_memory_per_node, worker_node_options=worker_node_options, collect_log_to_path=collect_log_to_path, autoscale_mode=True, spark_job_server_port=self.server.server_address[1])
    except Exception:
        if spark_job_group_id in self.server.task_status_dict:
            self.server.task_status_dict.pop(spark_job_group_id)
        _logger.warning(f'Spark job {spark_job_group_id} hosting Ray worker node exit.')