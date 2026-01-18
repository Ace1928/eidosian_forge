import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_get_model_info(self) -> None:
    hub.get_model_info('mnist', self.repo, opset=8)
    hub.get_model_info('mnist', self.repo)
    self.assertRaises(AssertionError, lambda: hub.get_model_info('mnist', self.repo, opset=-1))