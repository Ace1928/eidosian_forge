import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
@reader
def read_dftb_lattice(fileobj, images=None):
    """Read lattice vectors from MD and return them as a list.

    If a molecules are parsed add them there."""
    if images is not None:
        append = True
        if hasattr(images, 'get_positions'):
            images = [images]
    else:
        append = False
    fileobj.seek(0)
    lattices = []
    for line in fileobj:
        if 'Lattice vectors' in line:
            vec = []
            for i in range(3):
                line = fileobj.readline().split()
                try:
                    line = [float(x) for x in line]
                except ValueError:
                    raise ValueError('Lattice vector elements should be of type float.')
                vec.extend(line)
            lattices.append(np.array(vec).reshape((3, 3)))
    if append:
        if len(images) != len(lattices):
            raise ValueError('Length of images given does not match number of cell vectors found')
        for i, atoms in enumerate(images):
            atoms.set_cell(lattices[i])
            atoms.set_pbc(True)
        return
    else:
        return lattices