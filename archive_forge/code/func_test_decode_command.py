import os
import string
import sys
import unittest
from collections import defaultdict
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4
from shortuuid.cli import cli
from shortuuid.main import decode
from shortuuid.main import encode
from shortuuid.main import get_alphabet
from shortuuid.main import random
from shortuuid.main import set_alphabet
from shortuuid.main import ShortUUID
from shortuuid.main import uuid
@patch('shortuuid.cli.print')
def test_decode_command(self, mock_print):
    cli(['decode', 'CXc85b4rqinB7s5J52TRYb'])
    terminal_output = mock_print.call_args[0][0]
    self.assertEqual(terminal_output, '3b1f8b40-222c-4a6e-b77e-779d5a94e21c')