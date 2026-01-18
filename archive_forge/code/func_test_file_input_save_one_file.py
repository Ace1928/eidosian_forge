from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_file_input_save_one_file(document, comm, tmpdir):
    file_input = FileInput(accept='.txt')
    widget = file_input.get_root(document, comm=comm)
    assert isinstance(widget, BkFileInput)
    file_input._process_events({'mime_type': 'text/plain', 'value': 'U29tZSB0ZXh0Cg==', 'filename': 'testfile'})
    fpath = Path(tmpdir) / 'out.txt'
    file_input.save(str(fpath))
    assert fpath.exists()
    content = fpath.read_text()
    assert content == 'Some text\n'