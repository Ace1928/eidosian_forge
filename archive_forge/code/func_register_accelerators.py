from abc import ABC, abstractmethod
from typing import Any
import torch
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
@classmethod
def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
    pass