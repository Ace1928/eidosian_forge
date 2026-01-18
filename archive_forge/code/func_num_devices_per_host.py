import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
@staticmethod
def num_devices_per_host(device_type: str) -> int:
    return _get_device_handle(device_type).device_count()