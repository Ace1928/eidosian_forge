import base64
import os
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pytest
from jupyter_console.ptshell import ZMQTerminalInteractiveShell
def pil_called(data, mime):
    pil_called_with.append(data)