import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_attr_repeated_graph_proto(self) -> None:
    graphs = [GraphProto(), GraphProto()]
    graphs[0].name = 'a'
    graphs[1].name = 'b'
    attr = helper.make_attribute('graphs', graphs)
    self.assertEqual(attr.name, 'graphs')
    self.assertEqual(list(attr.graphs), graphs)
    checker.check_attribute(attr)