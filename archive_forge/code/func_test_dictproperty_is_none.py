import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_dictproperty_is_none():
    from kivy.properties import DictProperty
    d1 = DictProperty(None)
    d1.set_name(wid, 'd1')
    d1.link_eagerly(wid)
    assert d1.get(wid) is None
    d2 = DictProperty({'a': 1, 'b': 2}, allownone=True)
    d2.set_name(wid, 'd2')
    d2.link_eagerly(wid)
    d2.set(wid, None)
    assert d2.get(wid) is None