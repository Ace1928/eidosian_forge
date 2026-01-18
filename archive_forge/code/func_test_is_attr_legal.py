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
def test_is_attr_legal(self) -> None:
    attr = AttributeProto()
    self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
    attr = AttributeProto()
    attr.name = 'test'
    self.assertRaises(checker.ValidationError, checker.check_attribute, attr)
    attr = AttributeProto()
    attr.name = 'test'
    attr.f = 1.0
    attr.i = 2
    self.assertRaises(checker.ValidationError, checker.check_attribute, attr)