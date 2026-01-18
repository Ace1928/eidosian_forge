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
def test_no_frame_warning(tmpdir):
    fig, ax = plt.subplots()

    def update(frame):
        return []
    anim = animation.FuncAnimation(fig, update, frames=[], repeat=False, cache_frame_data=False)
    with pytest.warns(UserWarning, match='exhausted'):
        anim._start()