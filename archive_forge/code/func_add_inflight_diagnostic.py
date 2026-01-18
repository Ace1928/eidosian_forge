from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
@contextlib.contextmanager
def add_inflight_diagnostic(self, diagnostic: _Diagnostic) -> Generator[_Diagnostic, None, None]:
    """Adds a diagnostic to the context.

        Use this method to add diagnostics that are not created by the context.
        Args:
            diagnostic: The diagnostic to add.
        """
    self._inflight_diagnostics.append(diagnostic)
    try:
        yield diagnostic
    finally:
        self._inflight_diagnostics.pop()