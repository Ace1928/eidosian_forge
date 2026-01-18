import inspect
import numpy as np
import matplotlib as mpl
from matplotlib import (
from . import art3d, proj3d
@artist.allow_rasterization
def draw_grid(self, renderer):
    if not self.axes._draw_grid:
        return
    renderer.open_group('grid3d', gid=self.get_gid())
    ticks = self._update_ticks()
    if len(ticks):
        info = self._axinfo
        index = info['i']
        mins, maxs, _, _, _, highs = self._get_coord_info(renderer)
        minmax = np.where(highs, maxs, mins)
        maxmin = np.where(~highs, maxs, mins)
        xyz0 = np.tile(minmax, (len(ticks), 1))
        xyz0[:, index] = [tick.get_loc() for tick in ticks]
        lines = np.stack([xyz0, xyz0, xyz0], axis=1)
        lines[:, 0, index - 2] = maxmin[index - 2]
        lines[:, 2, index - 1] = maxmin[index - 1]
        self.gridlines.set_segments(lines)
        gridinfo = info['grid']
        self.gridlines.set_color(gridinfo['color'])
        self.gridlines.set_linewidth(gridinfo['linewidth'])
        self.gridlines.set_linestyle(gridinfo['linestyle'])
        self.gridlines.do_3d_projection()
        self.gridlines.draw(renderer)
    renderer.close_group('grid3d')