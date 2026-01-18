from __future__ import annotations
import dataclasses
from . import torch_wrapper
def ensure_immutable(self):
    assert self.storage.data_ptr() == self.ptr and self.storage.size() == self.size