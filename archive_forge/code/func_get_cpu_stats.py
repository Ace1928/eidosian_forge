from typing import Any, Dict, List, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.accelerators import _AcceleratorRegistry
from lightning_fabric.accelerators.cpu import _parse_cpu_cores
from lightning_fabric.utilities.types import _DEVICE
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
def get_cpu_stats() -> Dict[str, float]:
    if not _PSUTIL_AVAILABLE:
        raise ModuleNotFoundError(f'Fetching CPU device stats requires `psutil` to be installed. {str(_PSUTIL_AVAILABLE)}')
    import psutil
    return {_CPU_VM_PERCENT: psutil.virtual_memory().percent, _CPU_PERCENT: psutil.cpu_percent(), _CPU_SWAP_PERCENT: psutil.swap_memory().percent}