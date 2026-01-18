import sys
import threading
import unittest
import gi
from gi.repository import GObject, Gtk    # noqa: E402
from testtools import StreamToExtendedDecorator  # noqa: E402
from subunit import (PROGRESS_POP, PROGRESS_PUSH, PROGRESS_SET,  # noqa: E402
from subunit.progress_model import ProgressModel  # noqa: E402
def run_and_finish():
    test.run(result)
    result.stopTestRun()