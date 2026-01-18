import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import (
def test_nargs_cycler():
    from matplotlib.rcsetup import cycler as ccl
    with pytest.raises(TypeError, match='3 were given'):
        ccl(ccl(color=list('rgb')), 2, 3)