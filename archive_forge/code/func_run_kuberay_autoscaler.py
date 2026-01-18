import logging
import os
import subprocess
import time
import ray
from ray._private import ray_constants
from ray._private.ray_logging import setup_component_logger
from ray._private.services import get_node_ip_address
from ray._private.utils import try_to_create_directory
from ray.autoscaler._private.kuberay.autoscaling_config import AutoscalingConfigProducer
from ray.autoscaler._private.monitor import Monitor
def run_kuberay_autoscaler(cluster_name: str, cluster_namespace: str):
    """Wait until the Ray head container is ready. Then start the autoscaler."""
    head_ip = get_node_ip_address()
    ray_address = f'{head_ip}:6379'
    while True:
        try:
            subprocess.check_call(['ray', 'health-check', '--address', ray_address, '--skip-version-check'])
            print('The Ray head is ready. Starting the autoscaler.')
            break
        except subprocess.CalledProcessError:
            print('The Ray head is not yet ready.')
            print(f'Will check again in {BACKOFF_S} seconds.')
            time.sleep(BACKOFF_S)
    _setup_logging()
    autoscaling_config_producer = AutoscalingConfigProducer(cluster_name, cluster_namespace)
    Monitor(address=ray_address, autoscaling_config=autoscaling_config_producer, monitor_ip=head_ip, retry_on_failure=False).run()