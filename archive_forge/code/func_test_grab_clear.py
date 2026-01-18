import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_grab_clear():
    fig, ax = plt.subplots()
    fig.canvas.grab_mouse(ax)
    assert fig.canvas.mouse_grabber == ax
    fig.clear()
    assert fig.canvas.mouse_grabber is None