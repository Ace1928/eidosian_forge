import os
from typing import cast
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython import config
from curtsies.window import CursorAwareWindow
def process_multiple_events(self, event_list):
    for event in event_list:
        self.repl.process_event(event)