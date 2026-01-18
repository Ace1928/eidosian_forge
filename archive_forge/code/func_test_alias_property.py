import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('watch_before_use', [True, False])
def test_alias_property(self, watch_before_use):
    from kivy.properties import AliasProperty

    class CustomAlias(EventDispatcher):

        def _get_prop(self):
            self.getter_called += 1

        def _set_prop(self, value):
            self.setter_called += 1
        prop = AliasProperty(_get_prop, _set_prop, watch_before_use=watch_before_use)

        def __init__(self, **kwargs):
            super(CustomAlias, self).__init__(**kwargs)
            self.getter_called = 0
            self.setter_called = 0
            self.callback_called = 0

    def callback(widget, value):
        widget.callback_called += 1
    wid = CustomAlias()
    wid.bind(prop=callback)
    self.assertEqual(wid.getter_called, 0)
    self.assertEqual(wid.setter_called, 0)
    self.assertEqual(wid.callback_called, 0)
    wid.prop = 1
    self.assertEqual(wid.getter_called, 0)
    self.assertEqual(wid.setter_called, 1)
    self.assertEqual(wid.callback_called, 0)
    wid.prop = 1
    self.assertEqual(wid.getter_called, 0)
    self.assertEqual(wid.setter_called, 2)
    self.assertEqual(wid.callback_called, 0)
    self.assertEqual(wid.prop, None)
    self.assertEqual(wid.getter_called, 1)
    self.assertEqual(wid.setter_called, 2)
    self.assertEqual(wid.callback_called, 0)