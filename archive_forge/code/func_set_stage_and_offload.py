import base64
import json
import os
from copy import deepcopy
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
def set_stage_and_offload(self):
    self._stage = self.get_value('zero_optimization.stage', -1)
    self._offload = False
    if self.is_zero2() or self.is_zero3():
        offload_devices_valid = set(['cpu', 'nvme'])
        offload_devices = set([self.get_value('zero_optimization.offload_optimizer.device'), self.get_value('zero_optimization.offload_param.device')])
        if len(offload_devices & offload_devices_valid) > 0:
            self._offload = True