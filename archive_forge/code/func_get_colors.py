from math import cos, sin, sqrt
from os.path import basename
import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.geometry import complete_cell
from ase.gui.repeat import Repeat
from ase.gui.rotate import Rotate
from ase.gui.render import Render
from ase.gui.colors import ColorWindow
from ase.gui.utils import get_magmoms
from ase.utils import rotate
def get_colors(self, rgb=False):
    if rgb:
        return [tuple((int(_rgb[i:i + 2], 16) / 255 for i in range(1, 7, 2))) for _rgb in self.get_colors()]
    if self.colormode == 'jmol':
        return [self.colors.get(Z, BLACKISH) for Z in self.atoms.numbers]
    if self.colormode == 'neighbors':
        return [self.colors.get(Z, BLACKISH) for Z in self.get_color_scalars()]
    colorscale, cmin, cmax = self.colormode_data
    N = len(colorscale)
    colorswhite = colorscale + ['#ffffff']
    if cmin == cmax:
        indices = [N // 2] * len(self.atoms)
    else:
        scalars = np.ma.array(self.get_color_scalars())
        indices = np.clip(((scalars - cmin) / (cmax - cmin) * N + 0.5).astype(int), 0, N - 1)
    return [colorswhite[i] for i in indices.filled(N)]