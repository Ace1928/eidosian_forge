import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
@synchronized
def push_solid(self, function):
    """
        Push a function which performs gl commands
        used to build a display list. (The list is
        built outside of the function)
        """
    assert callable(function)
    self._draw_solid.append(function)
    if len(self._draw_solid) > self._max_render_stack_size:
        del self._draw_solid[1]