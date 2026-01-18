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
def test_movie_writer_dpi_default():

    class DummyMovieWriter(animation.MovieWriter):

        def _run(self):
            pass
    fig = plt.figure()
    filename = 'unused.null'
    fps = 5
    codec = 'unused'
    bitrate = 1
    extra_args = ['unused']
    writer = DummyMovieWriter(fps, codec, bitrate, extra_args)
    writer.setup(fig, filename)
    assert writer.dpi == fig.dpi