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
def set_device_pixel_ratio(ratio):
    p.return_value = ratio
    screen.logicalDotsPerInchChanged.emit(96)
    qt_canvas.draw()
    qt_canvas.flush_events()
    assert qt_canvas.device_pixel_ratio == ratio