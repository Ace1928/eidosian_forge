from abc import ABC, abstractmethod
from typing import Any
import torch
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
@staticmethod
@abstractmethod
def get_parallel_devices(devices: Any) -> Any:
    """Gets parallel devices for the Accelerator."""