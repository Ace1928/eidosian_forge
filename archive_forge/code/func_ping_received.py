import unittest
from unittest.mock import Mock
from IPython.core import events
import IPython.testing.tools as tt
@events._define_event
def ping_received():
    pass