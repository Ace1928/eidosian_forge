import unittest
from bpython import keys
def test_keymap_keyerror(self):
    """Verify keys.KeyMap raising KeyError when getting undefined key"""
    with self.assertRaises(KeyError):
        keys.urwid_key_dispatch['C-asdf']
        keys.urwid_key_dispatch['C-qwerty']