import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def calc_mem_ray_head_node(configured_object_store_bytes):
    import psutil
    if RAY_ON_SPARK_DRIVER_PHYSICAL_MEMORY_BYTES in os.environ:
        available_physical_mem = int(os.environ[RAY_ON_SPARK_DRIVER_PHYSICAL_MEMORY_BYTES])
    else:
        available_physical_mem = psutil.virtual_memory().total
    if RAY_ON_SPARK_DRIVER_SHARED_MEMORY_BYTES in os.environ:
        available_shared_mem = int(os.environ[RAY_ON_SPARK_DRIVER_SHARED_MEMORY_BYTES])
    else:
        available_shared_mem = psutil.virtual_memory().total
    heap_mem_bytes, object_store_bytes, warning_msg = _calc_mem_per_ray_node(available_physical_mem, available_shared_mem, configured_object_store_bytes)
    if warning_msg is not None:
        _logger.warning(warning_msg)
    return (heap_mem_bytes, object_store_bytes)