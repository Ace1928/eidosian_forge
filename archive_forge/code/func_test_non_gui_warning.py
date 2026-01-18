import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
@pytest.mark.backend('pdf')
def test_non_gui_warning(monkeypatch):
    plt.subplots()
    monkeypatch.setenv('DISPLAY', ':999')
    with pytest.warns(UserWarning) as rec:
        plt.show()
        assert len(rec) == 1
        assert 'FigureCanvasPdf is non-interactive, and thus cannot be shown' in str(rec[0].message)
    with pytest.warns(UserWarning) as rec:
        plt.gcf().show()
        assert len(rec) == 1
        assert 'FigureCanvasPdf is non-interactive, and thus cannot be shown' in str(rec[0].message)