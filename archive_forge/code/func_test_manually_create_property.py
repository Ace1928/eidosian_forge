import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
@pytest.mark.parametrize('by_val', [True, False])
def test_manually_create_property(by_val):
    from kivy.event import EventDispatcher
    from kivy.properties import StringProperty

    class Event(EventDispatcher):
        pass
    event = Event()
    assert not hasattr(event, 'a')
    if by_val:
        event.create_property('a', 'hello')
    else:
        event.apply_property(a=StringProperty('hello'))
    args = 0

    def callback(obj, val):
        nonlocal args
        args = (obj, val)
    event.fbind('a', callback)
    assert event.a == 'hello'
    event.a = 'bye'
    assert event.a == 'bye'
    assert args == (event, 'bye')
    event2 = Event()
    assert event2.a == 'hello'
    event2.fbind('a', callback)
    event2.a = 'goodbye'
    assert event2.a == 'goodbye'
    assert args == (event2, 'goodbye')