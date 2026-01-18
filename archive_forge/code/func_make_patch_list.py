import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
def make_patch_list(writer):
    from matplotlib.path import Path
    from matplotlib.patches import Circle, PathPatch, Wedge
    indices = writer.positions[:, 2].argsort()
    patch_list = []
    for a in indices:
        xy = writer.positions[a, :2]
        if a < writer.natoms:
            r = writer.d[a] / 2
            if writer.frac_occ:
                site_occ = writer.occs[str(writer.tags[a])]
                if np.sum([v for v in site_occ.values()]) < 1.0:
                    fill = '#ffffff'
                    patch = Circle(xy, r, facecolor=fill, edgecolor='black')
                    patch_list.append(patch)
                start = 0
                for sym, occ in sorted(site_occ.items(), key=lambda x: x[1], reverse=True):
                    if np.round(occ, decimals=4) == 1.0:
                        patch = Circle(xy, r, facecolor=writer.colors[a], edgecolor='black')
                        patch_list.append(patch)
                    else:
                        extent = 360.0 * occ
                        patch = Wedge(xy, r, start, start + extent, facecolor=jmol_colors[atomic_numbers[sym]], edgecolor='black')
                        patch_list.append(patch)
                        start += extent
            elif xy[1] + r > 0 and xy[1] - r < writer.h and (xy[0] + r > 0) and (xy[0] - r < writer.w):
                patch = Circle(xy, r, facecolor=writer.colors[a], edgecolor='black')
                patch_list.append(patch)
        else:
            a -= writer.natoms
            c = writer.T[a]
            if c != -1:
                hxy = writer.D[c]
                patch = PathPatch(Path((xy + hxy, xy - hxy)))
                patch_list.append(patch)
    return patch_list