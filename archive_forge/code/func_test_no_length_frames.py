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
@pytest.mark.parametrize('anim', [{'save_count': 10, 'frames': iter(range(5))}], indirect=['anim'])
def test_no_length_frames(anim):
    anim.save('unused.null', writer=NullMovieWriter())