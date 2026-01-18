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
@pytest.mark.parametrize('writer', [pytest.param('ffmpeg', marks=pytest.mark.skipif(not animation.FFMpegWriter.isAvailable(), reason='Requires FFMpeg')), pytest.param('imagemagick', marks=pytest.mark.skipif(not animation.ImageMagickWriter.isAvailable(), reason='Requires ImageMagick'))])
@pytest.mark.parametrize('html, want', [('none', None), ('html5', '<video width'), ('jshtml', '<script ')])
@pytest.mark.parametrize('anim', [dict(klass=dict)], indirect=['anim'])
def test_animation_repr_html(writer, html, want, anim):
    if platform.python_implementation() == 'PyPy':
        np.testing.break_cycles()
    if writer == 'imagemagick' and html == 'html5' and (not animation.FFMpegWriter.isAvailable()):
        pytest.skip('Requires FFMpeg')
    anim = animation.FuncAnimation(**anim)
    with plt.rc_context({'animation.writer': writer, 'animation.html': html}):
        html = anim._repr_html_()
    if want is None:
        assert html is None
        with pytest.warns(UserWarning):
            del anim
            np.testing.break_cycles()
    else:
        assert want in html