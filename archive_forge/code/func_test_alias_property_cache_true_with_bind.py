import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('watch_before_use', [True, False])
def test_alias_property_cache_true_with_bind(self, watch_before_use):
    from kivy.properties import NumericProperty, AliasProperty

    class CustomAlias(EventDispatcher):
        base_value = NumericProperty(1)

        def _get_prop(self):
            self.getter_called += 1
            return self.base_value * 2

        def _set_prop(self, value):
            self.base_value = value / 2
        prop = AliasProperty(_get_prop, _set_prop, bind=('base_value',), cache=True, watch_before_use=watch_before_use)

        def __init__(self, **kwargs):
            super(CustomAlias, self).__init__(**kwargs)
            self.getter_called = 0
    wid = CustomAlias()
    self.assertEqual(wid.getter_called, 0)
    self.assertEqual(wid.base_value, 1)
    self.assertEqual(wid.getter_called, 0)
    wid.base_value = 4
    self.assertEqual(wid.getter_called, int(watch_before_use))
    self.assertEqual(wid.prop, 8)
    self.assertEqual(wid.getter_called, 1)
    wid.prop = 4
    self.assertEqual(wid.getter_called, 2)
    self.assertEqual(wid.base_value, 2)
    self.assertEqual(wid.prop, 4)
    self.assertEqual(wid.getter_called, 2)