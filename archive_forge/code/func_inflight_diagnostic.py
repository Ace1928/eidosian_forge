from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def inflight_diagnostic(self, rule: Optional[infra.Rule]=None) -> _Diagnostic:
    if rule is None:
        if len(self._inflight_diagnostics) <= 0:
            raise AssertionError('No inflight diagnostics')
        return self._inflight_diagnostics[-1]
    else:
        for diagnostic in reversed(self._inflight_diagnostics):
            if diagnostic.rule == rule:
                return diagnostic
        raise AssertionError(f'No inflight diagnostic for rule {rule.name}')