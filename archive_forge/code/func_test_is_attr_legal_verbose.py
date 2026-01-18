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
def test_is_attr_legal_verbose(self) -> None:

    def _set(attr: AttributeProto, type_: AttributeProto.AttributeType, var: str, value: Any) -> None:
        setattr(attr, var, value)
        attr.type = type_

    def _extend(attr: AttributeProto, type_: AttributeProto.AttributeType, var: List[Any], value: Any) -> None:
        var.extend(value)
        attr.type = type_
    SET_ATTR = [lambda attr: _set(attr, AttributeProto.FLOAT, 'f', 1.0), lambda attr: _set(attr, AttributeProto.INT, 'i', 1), lambda attr: _set(attr, AttributeProto.STRING, 's', b'str'), lambda attr: _extend(attr, AttributeProto.FLOATS, attr.floats, [1.0, 2.0]), lambda attr: _extend(attr, AttributeProto.INTS, attr.ints, [1, 2]), lambda attr: _extend(attr, AttributeProto.STRINGS, attr.strings, [b'a', b'b'])]
    for _i in range(100):
        attr = AttributeProto()
        attr.name = 'test'
        random.choice(SET_ATTR)(attr)
        checker.check_attribute(attr)
    for _i in range(100):
        attr = AttributeProto()
        attr.name = 'test'
        for func in random.sample(SET_ATTR, 2):
            func(attr)
        self.assertRaises(checker.ValidationError, checker.check_attribute, attr)