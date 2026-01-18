import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
def is_event_type(self, key):
    return key.startswith('on_')