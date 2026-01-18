from ase.neighborlist import build_neighbor_list, get_distance_matrix, get_distance_indices
from ase.ga.utilities import get_rdf
from ase import Atoms
def get_bond_value(self, imIdx, idxs, mic=True, **kwargs):
    """Get bond length.

        Parameters:

        imIdx: int
            Index of Image to get value from.
        idxs: tuple or list of integers
            Get distance between atoms idxs[0]-idxs[1].
        mic: bool
            Passed on to :func:`ase.Atoms.get_distance` for retrieving the value, defaults to True.
            If the cell of the image is correctly set, there should be no reason to change this.
        kwargs: options or dict
            Passed on to :func:`ase.Atoms.get_distance`.

        Returns:

        return: float
            Value returned by image.get_distance.
        """
    return self.images[imIdx].get_distance(idxs[0], idxs[1], mic=mic, **kwargs)