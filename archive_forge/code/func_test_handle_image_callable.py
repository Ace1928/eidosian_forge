import base64
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pytest
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
def test_handle_image_callable(self):
    called_with = []
    self.shell.callable_image_handler = called_with.append
    self.shell.handle_image_callable(self.data, self.mime)
    self.assertEqual(len(called_with), 1)
    assert called_with[0] is self.data