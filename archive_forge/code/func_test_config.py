import unittest
from bpython.curtsiesfrontend.manual_readline import (
def test_config(self):

    def f(cursor_offset, line):
        return ('hi', 2)

    def g(cursor_offset, line):
        return ('hey', 3)
    self.edits.add_config_attr('att', f)
    self.assertNotIn('att', self.edits)

    class config:
        att = 'c'
    key_dispatch = {'c': 'c'}
    configured_edits = self.edits.mapping_with_config(config, key_dispatch)
    self.assertTrue(configured_edits.__contains__, 'c')
    self.assertNotIn('c', self.edits)
    with self.assertRaises(NotImplementedError):
        configured_edits.add_config_attr('att2', g)
    with self.assertRaises(NotImplementedError):
        configured_edits.add('d', g)
    self.assertEqual(configured_edits.call('c', cursor_offset=5, line='asfd'), ('hi', 2))