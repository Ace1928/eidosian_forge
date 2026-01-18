from ase.atoms import Atoms
from ase.collections import g2
def molecule(name, vacuum=None, **kwargs):
    """Create an atomic structure from a database.

    This is a helper function to easily create molecules from the g2 and
    extra databases.

    Parameters
    ----------
    name : str
        Name of the molecule to build.
    vacuum : float, optional
        Amount of vacuum to pad the molecule with on all sides.
    Additional keyword arguments (kwargs) can be supplied, which are passed
    to ase.Atoms.

    Returns
    -------
    ase.atoms.Atoms
        An ASE Atoms object corresponding to the specified molecule.

    Notes
    -----
    To see a list of allowed names, try:

    >>> from ase.collections import g2
    >>> print(g2.names)
    >>> from ase.build.molecule import extra
    >>> print(extra.keys())

    Examples
    --------
    >>> from ase.build import molecule
    >>> atoms = molecule('H2O')

    """
    if name in extra:
        kwargs.update(extra[name])
        mol = Atoms(**kwargs)
    else:
        mol = g2[name]
        if kwargs:
            mol = Atoms(mol, **kwargs)
    if vacuum is not None:
        mol.center(vacuum=vacuum)
    return mol