from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def get_input_descriptors(self) -> list[int]:
    """
        return the fd from our serial device so we get called
        on input and responses
        """
    return [self._device.fd]