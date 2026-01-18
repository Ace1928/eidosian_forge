from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def malloc_preprocess(self, device_id, size, mem_size):
    self._cretate_frame_tree(used_bytes=mem_size)