import logging
import math
import sys
from typing import List, Tuple, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np
import pandas as pd
def plot_structure(self) -> None:
    logging.info('Plotting 3D hexagonal structure')
    logging.debug('Starting structure plotting')
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111, projection='3d')
    color_map = plt.get_cmap('viridis')
    for layer, hexagons in self.structure.items():
        color = color_map(layer / self.layers)
        for hexagon in hexagons:
            self.hexagon_connections(hexagon, ax, color=color)
            xs, ys, zs = zip(*hexagon)
            ax.plot(xs, ys, zs, color=color)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('3D Hexagonal Structure')
    plt.tight_layout()
    plt.show()
    logging.debug('Structure plotting completed')