import unittest
from bpython import keys
def test_keymap_getitem(self):
    """Verify keys.KeyMap correctly looking up items."""
    self.assertEqual(keys.urwid_key_dispatch['F11'], 'f11')
    self.assertEqual(keys.urwid_key_dispatch['C-a'], 'ctrl a')
    self.assertEqual(keys.urwid_key_dispatch['M-a'], 'meta a')