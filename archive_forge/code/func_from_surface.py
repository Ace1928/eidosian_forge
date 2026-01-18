from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
@classmethod
def from_surface(cls, vertices, nvertex, name='Other'):
    """
        Creates a new BrainModelAxis axis describing the vertices on a surface

        Parameters
        ----------
        vertices : array_like
            indices of the vertices on the surface
        nvertex : int
            total number of vertices on the surface
        name : str
            Name of the brain structure (e.g. 'CortexLeft' or 'CortexRight')

        Returns
        -------
        BrainModelAxis which covers (part of) the surface
        """
    cifti_name = cls.to_cifti_brain_structure_name(name)
    return cls(cifti_name, vertex=vertices, nvertices={cifti_name: nvertex})