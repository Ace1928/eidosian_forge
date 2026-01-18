from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple
import torch
from torch import Tensor
@abstractmethod
def configure_payload(self) -> Dict[str, Any]:
    """Returns a request payload as a dictionary."""