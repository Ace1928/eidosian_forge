from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
def padder(self) -> PaddingContext:
    return _ANSIX923PaddingContext(self.block_size)