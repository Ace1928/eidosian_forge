import base64
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pytest
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
def test_handle_image_tempfile(self):
    self.check_handler_with_file('{file}', 'tempfile')