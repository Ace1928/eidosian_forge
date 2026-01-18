from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def log_source_exception(self, level: int, exception: Exception) -> None:
    """Logs a source exception within the diagnostic.

        Invokes `log_section` and `log` to log the exception in markdown section format.
        """
    self.source_exception = exception
    with self.log_section(level, 'Exception log'):
        self.log(level, '%s', formatter.lazy_format_exception(exception))