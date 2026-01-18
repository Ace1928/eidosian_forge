import os
import tempfile
import textwrap
import unittest
from bpython import config
def test_keybindings_use_other_default(self):
    struct = self.load_temp_config(textwrap.dedent('\n            [keyboard]\n            help = C-h\n            '))
    self.assertEqual(struct.help_key, 'C-h')
    self.assertEqual(struct.backspace_key, '')