from abc import ABC, abstractmethod
from ctypes import ArgumentError
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from transformers.utils import is_tf_available
from ..base import ExportConfig
def supports_quantization_approach(self, quantization_approach: QuantizationApproach) -> bool:
    supported_approaches = self.SUPPORTED_QUANTIZATION_APPROACHES
    if isinstance(supported_approaches, dict):
        supported_approaches = supported_approaches.get(self.task, supported_approaches['default'])
    return quantization_approach in supported_approaches