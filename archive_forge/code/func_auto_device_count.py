from abc import ABC, abstractmethod
from typing import Any
import torch
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
@staticmethod
@abstractmethod
def auto_device_count() -> int:
    """Get the device count when set to auto."""