import sys
from cupy.cuda import memory_hook
def alloc_postprocess(self, **kwargs):
    msg = '{"hook":"%s","device_id":%d,"mem_size":%d,"mem_ptr":%d}'
    msg %= ('alloc', kwargs['device_id'], kwargs['mem_size'], kwargs['mem_ptr'])
    self._print(msg)