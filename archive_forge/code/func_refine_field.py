import numpy as np
from matplotlib import _api
from matplotlib.tri._triangulation import Triangulation
import matplotlib.tri._triinterpolate
def refine_field(self, z, triinterpolator=None, subdiv=3):
    """
        Refine a field defined on the encapsulated triangulation.

        Parameters
        ----------
        z : (npoints,) array-like
            Values of the field to refine, defined at the nodes of the
            encapsulated triangulation. (``n_points`` is the number of points
            in the initial triangulation)
        triinterpolator : `~matplotlib.tri.TriInterpolator`, optional
            Interpolator used for field interpolation. If not specified,
            a `~matplotlib.tri.CubicTriInterpolator` will be used.
        subdiv : int, default: 3
            Recursion level for the subdivision.
            Each triangle is divided into ``4**subdiv`` child triangles.

        Returns
        -------
        refi_tri : `~matplotlib.tri.Triangulation`
             The returned refined triangulation.
        refi_z : 1D array of length: *refi_tri* node count.
             The returned interpolated field (at *refi_tri* nodes).
        """
    if triinterpolator is None:
        interp = matplotlib.tri.CubicTriInterpolator(self._triangulation, z)
    else:
        _api.check_isinstance(matplotlib.tri.TriInterpolator, triinterpolator=triinterpolator)
        interp = triinterpolator
    refi_tri, found_index = self.refine_triangulation(subdiv=subdiv, return_tri_index=True)
    refi_z = interp._interpolate_multikeys(refi_tri.x, refi_tri.y, tri_index=found_index)[0]
    return (refi_tri, refi_z)