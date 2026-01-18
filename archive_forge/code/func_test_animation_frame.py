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
@check_figures_equal(extensions=['png'])
def test_animation_frame(tmpdir, fig_test, fig_ref):
    ax = fig_test.add_subplot()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    x = np.linspace(0, 2 * np.pi, 100)
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(x, np.sin(x + i / 100))
        return (line,)
    anim = animation.FuncAnimation(fig_test, animate, init_func=init, frames=5, blit=True, repeat=False)
    with tmpdir.as_cwd():
        anim.save('test.gif')
    ax = fig_ref.add_subplot()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    ax.plot(x, np.sin(x + 4 / 100))