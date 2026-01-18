import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def monkhorst_pack_interpolate(path, values, icell, bz2ibz, size, offset=(0, 0, 0), pad_width=2):
    """Interpolate values from Monkhorst-Pack sampling.

    `monkhorst_pack_interpolate` takes a set of `values`, for example
    eigenvalues, that are resolved on a Monkhorst Pack k-point grid given by
    `size` and `offset` and interpolates the values onto the k-points
    in `path`.

    Note
    ----
    For the interpolation to work, path has to lie inside the domain
    that is spanned by the MP kpoint grid given by size and offset.

    To try to ensure this we expand the domain slightly by adding additional
    entries along the edges and sides of the domain with values determined by
    wrapping the values to the opposite side of the domain. In this way we
    assume that the function to be interpolated is a periodic function in
    k-space. The padding width is determined by the `pad_width` parameter.

    Parameters
    ----------
    path: (nk, 3) array-like
        Desired path in units of reciprocal lattice vectors.
    values: (nibz, ...) array-like
        Values on Monkhorst-Pack grid.
    icell: (3, 3) array-like
        Reciprocal lattice vectors.
    bz2ibz: (nbz,) array-like of int
        Map from nbz points in BZ to nibz reduced points in IBZ.
    size: (3,) array-like of int
        Size of Monkhorst-Pack grid.
    offset: (3,) array-like
        Offset of Monkhorst-Pack grid.
    pad_width: int
        Padding width to aid interpolation

    Returns
    -------
    (nbz,) array-like
        *values* interpolated to *path*.
    """
    from scipy.interpolate import LinearNDInterpolator
    path = (np.asarray(path) + 0.5) % 1 - 0.5
    path = np.dot(path, icell)
    v = np.asarray(values)[bz2ibz]
    v = v.reshape(tuple(size) + v.shape[1:])
    size = np.asarray(size)
    i = np.indices(size + 2 * pad_width).transpose((1, 2, 3, 0)).reshape((-1, 3))
    k = (i - pad_width + 0.5) / size - 0.5 + offset
    k = np.dot(k, icell)
    V = np.pad(v, [(pad_width, pad_width)] * 3 + [(0, 0)] * (v.ndim - 3), mode='wrap')
    interpolate = LinearNDInterpolator(k, V.reshape((-1,) + V.shape[3:]))
    interpolated_points = interpolate(path)
    assert not np.isnan(interpolated_points).any(), 'Points outside interpolation domain. Try increasing pad_width.'
    return interpolated_points