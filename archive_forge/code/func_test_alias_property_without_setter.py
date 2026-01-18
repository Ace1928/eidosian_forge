import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('watch_before_use', [True, False])
def test_alias_property_without_setter(self, watch_before_use):
    from kivy.properties import AliasProperty
    expected_value = 5

    class CustomAlias(EventDispatcher):

        def _get_prop(self):
            self.getter_called += 1
            return expected_value
        prop = AliasProperty(_get_prop, None, watch_before_use=watch_before_use)

        def __init__(self, **kwargs):
            super(CustomAlias, self).__init__(**kwargs)
            self.getter_called = 0
    wid = CustomAlias()
    self.assertEqual(wid.getter_called, 0)
    value = wid.prop
    self.assertEqual(value, expected_value)
    self.assertEqual(wid.getter_called, 1)
    self.assertRaises(AttributeError, partial(setattr, wid, 'prop', 1))