from ase.io.utils import PlottingVariables, make_patch_list
def plot_atoms(atoms, ax=None, **parameters):
    """Plot an atoms object in a matplotlib subplot.

    Parameters
    ----------
    atoms : Atoms object
    ax : Matplotlib subplot object
    rotation : str, optional
        In degrees. In the form '10x,20y,30z'
    show_unit_cell : int, optional, default 2
        Draw the unit cell as dashed lines depending on value:
        0: Don't
        1: Do
        2: Do, making sure cell is visible
    radii : float, optional
        The radii of the atoms
    colors : list of strings, optional
        Color of the atoms, must be the same length as
        the number of atoms in the atoms object.
    scale : float, optional
        Scaling of the plotted atoms and lines.
    offset : tuple (float, float), optional
        Offset of the plotted atoms and lines.
    """
    if isinstance(atoms, list):
        assert len(atoms) == 1
        atoms = atoms[0]
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    Matplotlib(atoms, ax, **parameters).write()
    return ax