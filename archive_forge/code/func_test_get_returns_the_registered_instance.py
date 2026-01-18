import os
import tempfile
import unittest
import onnx
def test_get_returns_the_registered_instance(self) -> None:
    serializer = onnx.serialization.registry.get('onnxtext')
    self.assertIs(serializer, self.serializer)