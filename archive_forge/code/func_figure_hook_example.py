import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def figure_hook_example(figure):
    figure._test_was_here = True