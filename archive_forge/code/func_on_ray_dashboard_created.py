import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time
def on_ray_dashboard_created(self, port):
    display_databricks_driver_proxy_url(get_spark_session().sparkContext, port, 'Ray Cluster Dashboard')