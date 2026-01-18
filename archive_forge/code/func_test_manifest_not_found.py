import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_manifest_not_found(self) -> None:
    self.assertRaises(AssertionError, lambda: hub.load(self.name, 'onnx/models:unknown', silent=True))