from ase.atoms import Atoms
from ase.utils import reader, writer
@reader
def read_crystal(fd):
    """Method to read coordinates form 'fort.34' files
    additionally read information about
    periodic boundary condition
    """
    lines = fd.readlines()
    atoms_pos = []
    anumber_list = []
    my_pbc = [False, False, False]
    mycell = []
    if float(lines[4]) != 1:
        raise ValueError('High symmetry geometry is not allowed.')
    if float(lines[1].split()[0]) < 500.0:
        cell = [float(c) for c in lines[1].split()]
        mycell.append(cell)
        my_pbc[0] = True
    else:
        mycell.append([1, 0, 0])
    if float(lines[2].split()[1]) < 500.0:
        cell = [float(c) for c in lines[2].split()]
        mycell.append(cell)
        my_pbc[1] = True
    else:
        mycell.append([0, 1, 0])
    if float(lines[3].split()[2]) < 500.0:
        cell = [float(c) for c in lines[3].split()]
        mycell.append(cell)
        my_pbc[2] = True
    else:
        mycell.append([0, 0, 1])
    natoms = int(lines[9].split()[0])
    for i in range(natoms):
        index = 10 + i
        anum = int(lines[index].split()[0]) % 100
        anumber_list.append(anum)
        position = [float(p) for p in lines[index].split()[1:]]
        atoms_pos.append(position)
    atoms = Atoms(positions=atoms_pos, numbers=anumber_list, cell=mycell, pbc=my_pbc)
    return atoms