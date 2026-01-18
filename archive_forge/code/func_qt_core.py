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
@pytest.fixture
def qt_core(request):
    qt_compat = pytest.importorskip('matplotlib.backends.qt_compat')
    QtCore = qt_compat.QtCore
    return QtCore