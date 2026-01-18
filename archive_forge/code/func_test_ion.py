import difflib
import numpy as np
import sys
from pathlib import Path
import pytest
import matplotlib as mpl
from matplotlib.testing import subprocess_run_for_testing
from matplotlib import pyplot as plt
def test_ion():
    plt.ioff()
    assert not mpl.is_interactive()
    with plt.ion():
        assert mpl.is_interactive()
    assert not mpl.is_interactive()
    plt.ion()
    assert mpl.is_interactive()
    with plt.ion():
        assert mpl.is_interactive()
    assert mpl.is_interactive()