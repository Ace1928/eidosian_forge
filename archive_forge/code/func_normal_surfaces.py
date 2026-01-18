import snappy
import FXrays
def normal_surfaces(self, algorithm='FXrays'):
    """
    All the vertex spun-normal surfaces in the current triangulation.

    >>> M = Manifold('m004')
    >>> M.normal_surfaces()    # doctest: +NORMALIZE_WHITESPACE
    [<Surface 0: [0, 0] [1, 2] (4, 1)>,
     <Surface 1: [0, 1] [1, 2] (4, -1)>,
     <Surface 2: [1, 2] [2, 1] (-4, -1)>,
     <Surface 3: [2, 2] [2, 1] (-4, 1)>]
    """
    if 'normal_surfaces' not in self._cache:
        eqns = self._normal_surface_equations()
        self._cache['normal_surfaces'] = [SpunSurface(self, qv, index=i) for i, qv in enumerate(eqns.vertex_solutions(algorithm))]
    return self._cache['normal_surfaces']