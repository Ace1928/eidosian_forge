import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_node(self) -> None:
    node = helper.make_node('Relu', ['X'], ['Y'], name='test')
    checker.check_node(node)