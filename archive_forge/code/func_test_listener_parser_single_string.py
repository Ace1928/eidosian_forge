from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_listener_parser_single_string(self):
    text = 'some_trait_name'
    parser = traits_listener.ListenerParser(text=text)
    expected = traits_listener.ListenerItem(name='some_trait_name', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    self.assertEqual(parser.listener, expected)