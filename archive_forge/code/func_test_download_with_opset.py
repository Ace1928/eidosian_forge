import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_download_with_opset(self) -> None:
    model = hub.load(self.name, self.repo, opset=8)
    self.assertIsInstance(model, ModelProto)