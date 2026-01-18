from typing import Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import core, drawings, types
from qiskit.visualization.pulse_v2.plotters.base_plotter import BasePlotter
from qiskit.visualization.utils import matplotlib_close_if_inline
def initialize_canvas(self):
    """Format appearance of matplotlib canvas."""
    self.ax.set_facecolor(self.canvas.formatter['color.background'])
    self.ax.set_yticklabels([])
    self.ax.yaxis.set_tick_params(left=False)