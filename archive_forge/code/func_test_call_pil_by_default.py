import base64
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pytest
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
def test_call_pil_by_default(self):
    pil_called_with = []

    def pil_called(data, mime):
        pil_called_with.append(data)

    def raise_if_called(*args, **kwds):
        assert False
    shell = self.shell
    shell.handle_image_PIL = pil_called
    shell.handle_image_stream = raise_if_called
    shell.handle_image_tempfile = raise_if_called
    shell.handle_image_callable = raise_if_called
    shell.handle_image(None, None)
    assert len(pil_called_with) == 1