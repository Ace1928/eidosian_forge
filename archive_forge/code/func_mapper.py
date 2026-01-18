import subprocess
import os
import sys
import random
import threading
import collections
import logging
import time
def mapper(_):
    try:
        return _get_avail_mem_per_ray_worker_node(num_cpus_per_node, num_gpus_per_node, object_store_memory_per_node)
    except Exception as e:
        import traceback
        trace_msg = '\n'.join(traceback.format_tb(e.__traceback__))
        return (-1, -1, repr(e) + trace_msg, None)