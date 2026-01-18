from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
def sarif_log(self) -> sarif.SarifLog:
    """Returns the SARIF Log object."""
    return sarif.SarifLog(version=sarif_version.SARIF_VERSION, schema_uri=sarif_version.SARIF_SCHEMA_LINK, runs=[self.sarif()])