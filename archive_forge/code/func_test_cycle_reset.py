import contextlib
from io import StringIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cycler import cycler
def test_cycle_reset():
    fig, ax = plt.subplots()
    prop0 = StringIO()
    prop1 = StringIO()
    prop2 = StringIO()
    with contextlib.redirect_stdout(prop0):
        plt.getp(ax.plot([1, 2], label='label')[0])
    ax.set_prop_cycle(linewidth=[10, 9, 4])
    with contextlib.redirect_stdout(prop1):
        plt.getp(ax.plot([1, 2], label='label')[0])
    assert prop1.getvalue() != prop0.getvalue()
    ax.set_prop_cycle(None)
    with contextlib.redirect_stdout(prop2):
        plt.getp(ax.plot([1, 2], label='label')[0])
    assert prop2.getvalue() == prop0.getvalue()