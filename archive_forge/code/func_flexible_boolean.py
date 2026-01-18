import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def flexible_boolean(input, default):
    if input in [True, False]:
        return input
    if input in ('f', 'F', 'false', 'False'):
        return False
    if input in ('t', 'T', 'true', 'True'):
        return True
    return default