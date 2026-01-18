import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
def get_flat_tri_mask(self, min_circle_ratio=0.01, rescale=True):
    """
        Eliminate excessively flat border triangles from the triangulation.

        Returns a mask *new_mask* which allows to clean the encapsulated
        triangulation from its border-located flat triangles
        (according to their :meth:`circle_ratios`).
        This mask is meant to be subsequently applied to the triangulation
        using `.Triangulation.set_mask`.
        *new_mask* is an extension of the initial triangulation mask
        in the sense that an initially masked triangle will remain masked.

        The *new_mask* array is computed recursively; at each step flat
        triangles are removed only if they share a side with the current mesh
        border. Thus, no new holes in the triangulated domain will be created.

        Parameters
        ----------
        min_circle_ratio : float, default: 0.01
            Border triangles with incircle/circumcircle radii ratio r/R will
            be removed if r/R < *min_circle_ratio*.
        rescale : bool, default: True
            If True, first, internally rescale (based on `scale_factors`) so
            that the (unmasked) triangles fit exactly inside a unit square
            mesh.  This rescaling accounts for the difference of scale which
            might exist between the 2 axis.

        Returns
        -------
        array of bool
            Mask to apply to encapsulated triangulation.
            All the initially masked triangles remain masked in the
            *new_mask*.

        Notes
        -----
        The rationale behind this function is that a Delaunay
        triangulation - of an unstructured set of points - sometimes contains
        almost flat triangles at its border, leading to artifacts in plots
        (especially for high-resolution contouring).
        Masked with computed *new_mask*, the encapsulated
        triangulation would contain no more unmasked border triangles
        with a circle ratio below *min_circle_ratio*, thus improving the
        mesh quality for subsequent plots or interpolation.
        """
    ntri = self._triangulation.triangles.shape[0]
    mask_bad_ratio = self.circle_ratios(rescale) < min_circle_ratio
    current_mask = self._triangulation.mask
    if current_mask is None:
        current_mask = np.zeros(ntri, dtype=bool)
    valid_neighbors = np.copy(self._triangulation.neighbors)
    renum_neighbors = np.arange(ntri, dtype=np.int32)
    nadd = -1
    while nadd != 0:
        wavefront = (np.min(valid_neighbors, axis=1) == -1) & ~current_mask
        added_mask = wavefront & mask_bad_ratio
        current_mask = added_mask | current_mask
        nadd = np.sum(added_mask)
        valid_neighbors[added_mask, :] = -1
        renum_neighbors[added_mask] = -1
        valid_neighbors = np.where(valid_neighbors == -1, -1, renum_neighbors[valid_neighbors])
    return np.ma.filled(current_mask, True)