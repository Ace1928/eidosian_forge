from datetime import datetime, timezone, timedelta
import platform
from unittest.mock import MagicMock
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.units as munits
from matplotlib.category import UnitData
import numpy as np
import pytest
def test_empty_default_limits(quantity_converter):
    munits.registry[Quantity] = quantity_converter
    fig, ax1 = plt.subplots()
    ax1.xaxis.update_units(Quantity([10], 'miles'))
    fig.draw_without_rendering()
    assert ax1.get_xlim() == (0, 100)
    ax1.yaxis.update_units(Quantity([10], 'miles'))
    fig.draw_without_rendering()
    assert ax1.get_ylim() == (0, 100)
    fig, ax = plt.subplots()
    ax.axhline(30)
    ax.plot(Quantity(np.arange(0, 3), 'miles'), Quantity(np.arange(0, 6, 2), 'feet'))
    fig.draw_without_rendering()
    assert ax.get_xlim() == (0, 2)
    assert ax.get_ylim() == (0, 30)
    fig, ax = plt.subplots()
    ax.axvline(30)
    ax.plot(Quantity(np.arange(0, 3), 'miles'), Quantity(np.arange(0, 6, 2), 'feet'))
    fig.draw_without_rendering()
    assert ax.get_xlim() == (0, 30)
    assert ax.get_ylim() == (0, 4)
    fig, ax = plt.subplots()
    ax.xaxis.update_units(Quantity([10], 'miles'))
    ax.axhline(30)
    fig.draw_without_rendering()
    assert ax.get_xlim() == (0, 100)
    assert ax.get_ylim() == (28.5, 31.5)
    fig, ax = plt.subplots()
    ax.yaxis.update_units(Quantity([10], 'miles'))
    ax.axvline(30)
    fig.draw_without_rendering()
    assert ax.get_ylim() == (0, 100)
    assert ax.get_xlim() == (28.5, 31.5)