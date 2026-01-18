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
@property
def process_index(self):
    """
        The index of the current process used.
        """
    requires_backends(self, ['torch'])
    if self.distributed_state is not None:
        return self.distributed_state.process_index
    elif is_sagemaker_mp_enabled():
        return smp.dp_rank() if not smp.state.cfg.prescaled_batch else smp.rdp_rank()
    return 0