import os
from .start_hook_base import RayOnSparkStartHook
from .utils import get_spark_session
import logging
import threading
import time
def auto_shutdown_watcher():
    auto_shutdown_millis = auto_shutdown_minutes * 60 * 1000
    while True:
        if ray_cluster_handler.is_shutdown:
            return
        idle_time = db_api_entry.getIdleTimeMillisSinceLastNotebookExecution()
        if idle_time > auto_shutdown_millis:
            from ray.util.spark import cluster_init
            with cluster_init._active_ray_cluster_rwlock:
                if ray_cluster_handler is cluster_init._active_ray_cluster:
                    cluster_init.shutdown_ray_cluster()
            return
        time.sleep(DATABRICKS_AUTO_SHUTDOWN_POLL_INTERVAL_SECONDS)