import contextlib
import functools
import inspect
import os
from platform import uname
from pathlib import Path
import shutil
import string
import sys
import warnings
from packaging.version import parse as parse_version
import matplotlib.style
import matplotlib.units
import matplotlib.testing
from matplotlib import _pylab_helpers, cbook, ft2font, pyplot as plt, ticker
from .compare import comparable_formats, compare_images, make_test_filename
from .exceptions import ImageComparisonFailure
def copy_baseline(self, baseline, extension):
    baseline_path = self.baseline_dir / baseline
    orig_expected_path = baseline_path.with_suffix(f'.{extension}')
    if extension == 'eps' and (not orig_expected_path.exists()):
        orig_expected_path = orig_expected_path.with_suffix('.pdf')
    expected_fname = make_test_filename(self.result_dir / orig_expected_path.name, 'expected')
    try:
        with contextlib.suppress(OSError):
            os.remove(expected_fname)
        try:
            if 'microsoft' in uname().release.lower():
                raise OSError
            os.symlink(orig_expected_path, expected_fname)
        except OSError:
            shutil.copyfile(orig_expected_path, expected_fname)
    except OSError as err:
        raise ImageComparisonFailure(f'Missing baseline image {expected_fname} because the following file cannot be accessed: {orig_expected_path}') from err
    return expected_fname