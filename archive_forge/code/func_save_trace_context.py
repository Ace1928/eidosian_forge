from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
@abc.abstractmethod
def save_trace_context(self, trace_id: str, span_id: str, is_sampled: bool) -> None:
    """Saves the trace_id and span_id related to the current span.

        After register the plugin, if tracing is enabled, this method will be
        called after the server finished sending response.

        This method can be used to propagate census context.

        Args:
        trace_id: The identifier for the trace associated with the span as a
         32-character hexadecimal encoded string,
         e.g. 26ed0036f2eff2b7317bccce3e28d01f
        span_id: The identifier for the span as a 16-character hexadecimal encoded
         string. e.g. 113ec879e62583bc
        is_sampled: A bool indicates whether the span is sampled.
        """
    raise NotImplementedError()