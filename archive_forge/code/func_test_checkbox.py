from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_checkbox(document, comm):
    checkbox = Checkbox(value=True, name='Checkbox')
    widget = checkbox.get_root(document, comm=comm)
    assert isinstance(widget, checkbox._widget_type)
    assert widget.label == 'Checkbox'
    assert widget.active == True
    widget.active = False
    checkbox._process_events({'active': False})
    assert checkbox.value == False
    checkbox.value = True
    assert widget.active == True