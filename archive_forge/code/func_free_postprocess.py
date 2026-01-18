import sys
from cupy.cuda import memory_hook
def free_postprocess(self, **kwargs):
    msg = '{"hook":"%s","device_id":%d,"mem_size":%d,"mem_ptr":%d,"pmem_id":"%s"}'
    msg %= ('free', kwargs['device_id'], kwargs['mem_size'], kwargs['mem_ptr'], hex(kwargs['pmem_id']))
    self._print(msg)