import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('return_value', [None, 'string', 1, ('string',), 'artist'])
def test_draw_frame(return_value):
    fig, ax = plt.subplots()
    line, = ax.plot([])

    def animate(i):
        line.set_data([0, 1], [0, i])
        if return_value == 'artist':
            return line
        else:
            return return_value
    with pytest.raises(RuntimeError):
        animation.FuncAnimation(fig, animate, blit=True, cache_frame_data=False)