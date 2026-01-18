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
@parameterized.parameterized.expand([(AttributeProto.AttributeType.FLOAT, 'FLOAT'), (AttributeProto.AttributeType.INT, 'INT'), (AttributeProto.AttributeType.STRING, 'STRING'), (AttributeProto.AttributeType.TENSOR, 'TENSOR'), (AttributeProto.AttributeType.GRAPH, 'GRAPH'), (AttributeProto.AttributeType.SPARSE_TENSOR, 'SPARSE_TENSOR'), (AttributeProto.AttributeType.TYPE_PROTO, 'TYPE_PROTO'), (AttributeProto.AttributeType.FLOATS, 'FLOATS'), (AttributeProto.AttributeType.INTS, 'INTS'), (AttributeProto.AttributeType.STRINGS, 'STRINGS'), (AttributeProto.AttributeType.TENSORS, 'TENSORS'), (AttributeProto.AttributeType.GRAPHS, 'GRAPHS'), (AttributeProto.AttributeType.SPARSE_TENSORS, 'SPARSE_TENSORS'), (AttributeProto.AttributeType.TYPE_PROTOS, 'TYPE_PROTOS')])
def test_attr_type_to_str(self, attr_type, expected_str):
    result = helper._attr_type_to_str(attr_type)
    self.assertEqual(result, expected_str)