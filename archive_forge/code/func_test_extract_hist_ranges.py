import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_extract_hist_ranges():
    instr = '1 2/3 ~4/5-6 ~4/7-~4/9 ~9/2-~7/5 ~10/'
    expected = [(0, 1, 2), (2, 3, 4), (-4, 5, 7), (-4, 7, 10), (-9, 2, None), (-8, 1, None), (-7, 1, 6), (-10, 1, None)]
    actual = list(extract_hist_ranges(instr))
    assert actual == expected