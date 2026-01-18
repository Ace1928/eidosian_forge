import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('kwargs,error,message', [({'location': 'top', 'orientation': 'vertical'}, TypeError, 'location and orientation are mutually exclusive'), ({'location': 'top', 'orientation': 'vertical', 'cax': True}, TypeError, 'location and orientation are mutually exclusive'), ({'ticklocation': 'top', 'orientation': 'vertical', 'cax': True}, ValueError, "'top' is not a valid value for position"), ({'location': 'top', 'extendfrac': (0, None)}, ValueError, 'invalid value for extendfrac')])
def test_colorbar_errors(kwargs, error, message):
    fig, ax = plt.subplots()
    im = ax.imshow([[0, 1], [2, 3]])
    if kwargs.get('cax', None) is True:
        kwargs['cax'] = ax.inset_axes([0, 1.05, 1, 0.05])
    with pytest.raises(error, match=message):
        fig.colorbar(im, **kwargs)