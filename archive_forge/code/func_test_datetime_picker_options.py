from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_datetime_picker_options(document, comm):
    options = [datetime(2018, 9, 1), datetime(2018, 9, 2), datetime(2018, 9, 3)]
    datetime_picker = DatetimePicker(name='DatetimePicker', value=datetime(2018, 9, 2, 1, 5), options=options)
    assert datetime_picker.value == datetime(2018, 9, 2, 1, 5)
    assert datetime_picker.enabled_dates == options