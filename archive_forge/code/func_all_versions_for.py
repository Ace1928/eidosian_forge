from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def all_versions_for(op_name: str) -> list[tuple[str, int]]:
    domain, versions_set = ALL_OP_VERSIONS[op_name]
    if not versions_set:
        raise ValueError(f'No versions available for operator {op_name}')
    versions = sorted(versions_set)
    return [(f'version{version}', version) for version in versions if version > 5 or domain != ONNX_DOMAIN]