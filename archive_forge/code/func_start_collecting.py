import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
def start_collecting(self):
    self.obj.on_trait_change(self._event_handler, self.trait_name)