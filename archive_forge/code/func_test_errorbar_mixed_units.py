from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
def test_errorbar_mixed_units():
    x = np.arange(10)
    y = [datetime(2020, 5, i * 2 + 1) for i in x]
    fig, ax = plt.subplots()
    ax.errorbar(x, y, timedelta(days=0.5))
    fig.canvas.draw()