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
def hexagon_connections(self, hexagon: Hexagon3D, ax: plt.Axes, color: str) -> None:
    logging.debug(f'Drawing connections for hexagon with center={np.mean(np.array(hexagon), axis=0)}')
    for i in range(6):
        start = hexagon[i]
        for j in [1, 2, 3]:
            end = hexagon[(i + j) % 6]
            ax.add_artist(Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], mutation_scale=10, lw=1, arrowstyle='-|>', color=color))
    center: Point3D = np.mean(np.array(hexagon), axis=0)
    for vertex in hexagon:
        ax.add_artist(Arrow3D([vertex[0], center[0]], [vertex[1], center[1]], [vertex[2], center[2]], mutation_scale=10, lw=1, arrowstyle='-|>', color=color))
    logging.debug('Connections drawn')