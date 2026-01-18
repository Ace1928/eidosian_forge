from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def pop_inflight_diagnostic(self) -> _Diagnostic:
    """Pops the last diagnostic from the inflight diagnostics stack.

        Returns:
            The popped diagnostic.
        """
    return self._inflight_diagnostics.pop()