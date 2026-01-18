import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def handle_pick(event):
    assert event.mouseevent.key == 'a'
    picks.append(event)