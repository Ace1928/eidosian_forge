import logging
import os
import platform
import sys
import time
import ray  # noqa F401
import psutil  # noqa E402
def raise_if_low_memory(self):
    if self.disabled:
        return
    if time.time() - self.last_checked > self.check_interval:
        self.last_checked = time.time()
        used_gb, total_gb = self.get_memory_usage()
        if used_gb > total_gb * self.error_threshold:
            raise RayOutOfMemoryError(RayOutOfMemoryError.get_message(used_gb, total_gb, self.error_threshold))
        else:
            logger.debug(f'Memory usage is {used_gb} / {total_gb}')