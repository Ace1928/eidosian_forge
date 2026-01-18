import logging
import os
import platform
import sys
import time
import ray  # noqa F401
import psutil  # noqa E402
def get_shared(virtual_memory):
    """Get the estimated shared memory usage from psutil virtual mem info."""
    if hasattr(virtual_memory, 'shared'):
        return virtual_memory.shared
    else:
        return 0