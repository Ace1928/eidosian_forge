import os
import subprocess
import sys
import pytest
import matplotlib as mpl
import matplotlib.testing.compare
from matplotlib import pyplot as plt
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
@pytest.mark.parametrize('fmt, string', [('pdf', b'/CreationDate (D:20000101000000Z)'), ('ps', b'%%CreationDate: Sat Jan 01 00:00:00 2000')])
def test_determinism_source_date_epoch(fmt, string):
    """
    Test SOURCE_DATE_EPOCH support. Output a document with the environment
    variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the
    document contains the timestamp that corresponds to this date (given as an
    argument).

    Parameters
    ----------
    fmt : {"pdf", "ps", "svg"}
        Output format.
    string : bytes
        Timestamp string for 2000-01-01 00:00 UTC.
    """
    buf = subprocess.check_output([sys.executable, '-R', '-c', f"from matplotlib.tests.test_determinism import _save_figure; _save_figure('', {fmt!r})"], env={**os.environ, 'SOURCE_DATE_EPOCH': '946684800', 'MPLBACKEND': 'Agg'})
    assert string in buf