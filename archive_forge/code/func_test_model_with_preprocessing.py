import glob
import os
import unittest
from os.path import join
import pytest
from onnx import ModelProto, hub
def test_model_with_preprocessing(self) -> None:
    model = hub.load_composite_model('ResNet50-fp32', preprocessing_model='ResNet-preproc')
    self.assertIsInstance(model, ModelProto)