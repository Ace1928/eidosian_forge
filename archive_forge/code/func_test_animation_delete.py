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
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_animation_delete(anim):
    if platform.python_implementation() == 'PyPy':
        np.testing.break_cycles()
    anim = animation.FuncAnimation(**anim)
    with pytest.warns(Warning, match='Animation was deleted'):
        del anim
        np.testing.break_cycles()