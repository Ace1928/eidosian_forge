from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
def set_tracing(self, enable: bool) -> None:
    """Enable or disable tracing.

        Args:
        enable: A bool indicates whether tracing should be enabled.
        """
    self._tracing_enabled = enable