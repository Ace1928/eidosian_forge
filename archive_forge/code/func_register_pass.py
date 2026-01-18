from abc import ABC, abstractmethod
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Optional, Set, Tuple, Union
from onnx import ModelProto, load_model
from onnxruntime.transformers.onnx_model import OnnxModel
def register_pass(self, target: PreprocessorPass):
    if target not in self._passes:
        self._passes.append(target)