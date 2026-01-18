import os
import shutil
import tempfile
import threading
import unittest
from traits.api import HasTraits, on_trait_change, Bool, Float, List
from traits import trait_notifiers
from traits.util.event_tracer import (
def test_change_event_recorder(self):
    test_object = TestObject()
    container = RecordContainer()
    recorder = ChangeEventRecorder(container=container)
    trait_notifiers.set_change_event_tracers(pre_tracer=recorder.pre_tracer, post_tracer=recorder.post_tracer)
    try:
        test_object.number = 5.0
    finally:
        trait_notifiers.clear_change_event_tracers()
    filename = os.path.join(self.directory, 'MainThread.trace')
    container.save_to_file(filename)
    with open(filename, 'r', encoding='utf-8') as handle:
        lines = handle.readlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue("-> 'number' changed from 2.0 to 5.0 in 'TestObject'\n" in lines[0])
        self.assertTrue('CALLING' in lines[1])
        self.assertTrue('EXIT' in lines[2])