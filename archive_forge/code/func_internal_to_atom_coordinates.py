from Bio.PDB.Entity import Entity
def internal_to_atom_coordinates(self, verbose: bool=False) -> None:
    """Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems

        :raises Exception: if any chain does not have .internal_coord attribute
        """
    for chn in self.get_chains():
        chn.internal_to_atom_coordinates(verbose)