from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def test_context_assignment_different_thread(self):
    import threading
    VAR_VAR.set(None)
    ctx = Context()
    is_running = threading.Event()
    should_suspend = threading.Event()
    did_suspend = threading.Event()
    should_exit = threading.Event()
    holder = []

    def greenlet_in_thread_fn():
        VAR_VAR.set(1)
        is_running.set()
        should_suspend.wait(10)
        VAR_VAR.set(2)
        getcurrent().parent.switch()
        holder.append(VAR_VAR.get())

    def thread_fn():
        gr = greenlet(greenlet_in_thread_fn)
        gr.gr_context = ctx
        holder.append(gr)
        gr.switch()
        did_suspend.set()
        should_exit.wait(10)
        gr.switch()
        del gr
        greenlet()
    thread = threading.Thread(target=thread_fn, daemon=True)
    thread.start()
    is_running.wait(10)
    gr = holder[0]
    with self.assertRaisesRegex(ValueError, 'running in a different'):
        getattr(gr, 'gr_context')
    with self.assertRaisesRegex(ValueError, 'running in a different'):
        gr.gr_context = None
    should_suspend.set()
    did_suspend.wait(10)
    self.assertIs(gr.gr_context, ctx)
    self.assertEqual(gr.gr_context[VAR_VAR], 2)
    gr.gr_context = None
    should_exit.set()
    thread.join(10)
    self.assertEqual(holder, [gr, None])
    self.assertIsNone(gr.gr_context)
    gr.gr_context = ctx
    self.assertIs(gr.gr_context, ctx)
    del holder[:]
    gr = None
    thread = None