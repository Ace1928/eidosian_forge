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
@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_form_widget_get_with_datetime_and_date_fields():
    from matplotlib.backends.backend_qt import _create_qApp
    _create_qApp()
    form = [('Datetime field', datetime(year=2021, month=3, day=11)), ('Date field', date(year=2021, month=3, day=11))]
    widget = _formlayout.FormWidget(form)
    widget.setup()
    values = widget.get()
    assert values == [datetime(year=2021, month=3, day=11), date(year=2021, month=3, day=11)]