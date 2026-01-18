import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from traitlets.config.loader import Config
from IPython.core.history import HistoryAccessor, HistoryManager, extract_hist_ranges
def test_magic_rerun():
    """Simple test for %rerun (no args -> rerun last line)"""
    ip = get_ipython()
    ip.run_cell('a = 10', store_history=True)
    ip.run_cell('a += 1', store_history=True)
    assert ip.user_ns['a'] == 11
    ip.run_cell('%rerun', store_history=True)
    assert ip.user_ns['a'] == 12