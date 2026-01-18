from tempfile import TemporaryFile
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
from matplotlib import dviread
from matplotlib.testing import _has_tex_package
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('fmt', ['pdf', 'svg'])
def test_missing_psfont(fmt, monkeypatch):
    """An error is raised if a TeX font lacks a Type-1 equivalent"""
    monkeypatch.setattr(dviread.PsfontsMap, '__getitem__', lambda self, k: dviread.PsFont(texname=b'texfont', psname=b'Some Font', effects=None, encoding=None, filename=None))
    mpl.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'hello')
    with TemporaryFile() as tmpfile, pytest.raises(ValueError):
        fig.savefig(tmpfile, format=fmt)