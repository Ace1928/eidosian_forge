import unittest
import pytest
from kivy.event import EventDispatcher
from functools import partial
def test_inherit_property():
    from kivy.event import EventDispatcher
    from kivy.properties import StringProperty

    class Event(EventDispatcher):
        a = StringProperty('hello')

    class Event2(Event):
        b = StringProperty('hello2')
    event = Event2()
    args = 0

    def callback(obj, val):
        nonlocal args
        args = (obj, val)
    event.fbind('a', callback)
    event.fbind('b', callback)
    assert event.a == 'hello'
    assert event.b == 'hello2'
    event.a = 'bye'
    assert event.a == 'bye'
    assert args == (event, 'bye')
    event.b = 'goodbye'
    assert event.b == 'goodbye'
    assert args == (event, 'goodbye')