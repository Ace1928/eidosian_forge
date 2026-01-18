from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def getFromMatplotlib(name):
    """ 
    Generates a ColorMap object from a Matplotlib definition.
    Same as ``colormap.get(name, source='matplotlib')``.
    """
    try:
        import matplotlib.pyplot as mpl_plt
    except ModuleNotFoundError:
        return None
    cmap = None
    col_map = mpl_plt.get_cmap(name)
    if hasattr(col_map, '_segmentdata'):
        data = col_map._segmentdata
        if 'red' in data and isinstance(data['red'], (Sequence, np.ndarray)):
            positions = set()
            for key in ['red', 'green', 'blue']:
                for tup in data[key]:
                    positions.add(tup[0])
            col_data = np.zeros((len(positions), 4))
            col_data[:, -1] = sorted(positions)
            for idx, key in enumerate(['red', 'green', 'blue']):
                positions = np.zeros(len(data[key]))
                comp_vals = np.zeros(len(data[key]))
                for idx2, tup in enumerate(data[key]):
                    positions[idx2] = tup[0]
                    comp_vals[idx2] = tup[1]
                col_data[:, idx] = np.interp(col_data[:, 3], positions, comp_vals)
            cmap = ColorMap(pos=col_data[:, -1], color=255 * col_data[:, :3] + 0.5)
        elif 'red' in data and isinstance(data['red'], Callable):
            col_data = np.zeros((64, 4))
            col_data[:, -1] = np.linspace(0.0, 1.0, 64)
            for idx, key in enumerate(['red', 'green', 'blue']):
                col_data[:, idx] = np.clip(data[key](col_data[:, -1]), 0, 1)
            cmap = ColorMap(pos=col_data[:, -1], color=255 * col_data[:, :3] + 0.5)
    elif hasattr(col_map, 'colors'):
        col_data = np.array(col_map.colors)
        cmap = ColorMap(name=name, pos=np.linspace(0.0, 1.0, col_data.shape[0]), color=255 * col_data[:, :3] + 0.5)
    if cmap is not None:
        cmap.name = name
        _mapCache[name] = cmap
    return cmap