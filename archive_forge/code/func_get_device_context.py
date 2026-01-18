import threading
from typing import Any, Dict
import torch._C._lazy
def get_device_context(device=None):
    if device is None:
        device = torch._C._lazy._get_default_device_type()
    else:
        device = str(device)
    with DeviceContext._CONTEXTS_LOCK:
        devctx = DeviceContext._CONTEXTS.get(device, None)
        if devctx is None:
            devctx = DeviceContext(device)
            DeviceContext._CONTEXTS[device] = devctx
        return devctx