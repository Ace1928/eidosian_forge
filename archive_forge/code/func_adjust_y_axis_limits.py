import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
def adjust_y_axis_limits(ax, data, margin_factor=0.1):
    """
    Adjusts the y-axis limits dynamically based on the range of the data currently being visualized.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to adjust.
    data (np.ndarray): The data currently being visualized.
    margin_factor (float): A factor to determine the margin around the data range for better visibility.
    """
    y_min, y_max = (np.min(data), np.max(data))
    y_margin = (y_max - y_min) * margin_factor
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    logging.debug(f'Y-axis for ax set to {y_min - y_margin}, {y_max + y_margin}')