import os
import tempfile
import textwrap
import unittest
from bpython import config
def test_keybindings_use_other_default_issue_447(self):
    struct = self.load_temp_config(textwrap.dedent('\n            [keyboard]\n            help = F2\n            show_source = F9\n            '))
    self.assertEqual(struct.help_key, 'F2')
    self.assertEqual(struct.show_source_key, 'F9')