import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def test_global_register(self):
    ut = self

    class MyListener(ev.Listener):

        def on_start(self, event):
            ut.assertEqual(event.status, ev.EventStatus.START)
            ut.assertEqual(event.kind, 'numba:compile')
            dispatcher = event.data['dispatcher']
            ut.assertIs(dispatcher, foo)
            ut.assertNotIn(event.data['args'], dispatcher.overloads)

        def on_end(self, event):
            ut.assertEqual(event.status, ev.EventStatus.END)
            ut.assertEqual(event.kind, 'numba:compile')
            dispatcher = event.data['dispatcher']
            ut.assertIs(dispatcher, foo)
            ut.assertIn(event.data['args'], dispatcher.overloads)

    @njit
    def foo(x):
        return x
    listener = MyListener()
    ev.register('numba:compile', listener)
    foo(1)
    ev.unregister('numba:compile', listener)