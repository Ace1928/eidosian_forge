import os
import tempfile
import textwrap
import unittest
from bpython import config
def test_keybindings_unset(self):
    struct = self.load_temp_config(textwrap.dedent('\n            [keyboard]\n            help =\n            '))
    self.assertFalse(struct.help_key)