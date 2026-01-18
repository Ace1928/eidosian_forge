from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
def set_stats(self, enable: bool) -> None:
    """Enable or disable stats(metrics).

        Args:
        enable: A bool indicates whether stats should be enabled.
        """
    self._stats_enabled = enable