import copy
import importlib
import os
import signal
import sys
from datetime import date, datetime
from unittest import mock
import pytest
import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils
def fire_signal_and_quit():
    nonlocal event_loop_handler
    event_loop_handler = signal.getsignal(signal.SIGINT)
    qt_core.QCoreApplication.exit()