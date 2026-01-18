import unittest
from collections import namedtuple
from bpython.curtsies import combined_events
from bpython.test import FixLanguageTestCase as TestCase
import curtsies.events
def schedule_event(self, event, when):
    self._events.append(ScheduledEvent(when, event))
    self._events.sort()