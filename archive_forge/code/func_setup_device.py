from abc import ABC, abstractmethod
from typing import Any
import torch
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
@abstractmethod
def setup_device(self, device: torch.device) -> None:
    """Create and prepare the device for the current process."""