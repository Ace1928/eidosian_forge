from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_listener_handler_for_method(self):

    class A:

        def __init__(self, value):
            self.value = value

        def square(self):
            return self.value * self.value
    a = A(7)
    listener_handler = traits_listener.ListenerHandler(a.square)
    handler = listener_handler()
    self.assertEqual(handler(), 49)
    del a, handler
    handler = listener_handler()
    self.assertEqual(handler, Undefined)