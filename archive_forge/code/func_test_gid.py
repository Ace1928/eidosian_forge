import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
def test_gid():
    """Test that object gid appears in output svg."""
    from matplotlib.offsetbox import OffsetBox
    from matplotlib.axis import Tick
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.imshow([[1.0, 2.0], [2.0, 3.0]], aspect='auto')
    ax1.scatter([1, 2, 3], [1, 2, 3], label='myscatter')
    ax1.plot([2, 3, 1], label='myplot')
    ax1.legend()
    ax1a = ax1.twinx()
    ax1a.bar([1, 2, 3], [1, 2, 3])
    ax2 = fig.add_subplot(132, projection='polar')
    ax2.plot([0, 1.5, 3], [1, 2, 3])
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot([1, 2], [1, 2], [1, 2])
    fig.canvas.draw()
    gdic = {}
    for idx, obj in enumerate(fig.findobj(include_self=True)):
        if obj.get_visible():
            gid = f'test123{obj.__class__.__name__}_{idx}'
            gdic[gid] = obj
            obj.set_gid(gid)
    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue().decode()

    def include(gid, obj):
        if isinstance(obj, OffsetBox):
            return False
        if isinstance(obj, Text):
            if obj.get_text() == '':
                return False
            elif obj.axes is None:
                return False
        if isinstance(obj, plt.Line2D):
            xdata, ydata = obj.get_data()
            if len(xdata) == len(ydata) == 1:
                return False
            elif not hasattr(obj, 'axes') or obj.axes is None:
                return False
        if isinstance(obj, Tick):
            loc = obj.get_loc()
            if loc == 0:
                return False
            vi = obj.get_view_interval()
            if loc < min(vi) or loc > max(vi):
                return False
        return True
    for gid, obj in gdic.items():
        if include(gid, obj):
            assert gid in buf