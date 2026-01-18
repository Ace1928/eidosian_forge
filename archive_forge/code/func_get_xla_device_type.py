import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def get_xla_device_type(device: 'torch.device') -> Optional[str]:
    """
    Returns the xla device type (CPU|GPU|TPU) or None if the device is a non-xla device.
    """
    if is_torch_tpu_available():
        return xm.xla_real_devices([device])[0].split(':')[0]
    return None