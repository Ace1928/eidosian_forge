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
def get_available_versions(schema: OpSchema) -> set[int]:
    versions: set[int] = set()
    for version in range(schema.since_version, 0, -1):
        try:
            versions.add(defs.get_schema(schema.name, version, schema.domain).since_version)
        except SchemaError:
            break
    return versions