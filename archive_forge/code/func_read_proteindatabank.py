import warnings
import numpy as np
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from ase.io.espresso import label_to_symbol
from ase.utils import reader, writer
@reader
def read_proteindatabank(fileobj, index=-1, read_arrays=True):
    """Read PDB files."""
    images = []
    orig = np.identity(3)
    trans = np.zeros(3)
    occ = []
    bfactor = []
    residuenames = []
    residuenumbers = []
    atomtypes = []
    symbols = []
    positions = []
    cell = None
    pbc = None

    def build_atoms():
        atoms = Atoms(symbols=symbols, cell=cell, pbc=pbc, positions=positions)
        if not read_arrays:
            return atoms
        info = {'occupancy': occ, 'bfactor': bfactor, 'residuenames': residuenames, 'atomtypes': atomtypes, 'residuenumbers': residuenumbers}
        for name, array in info.items():
            if len(array) == 0:
                pass
            elif len(array) != len(atoms):
                warnings.warn('Length of {} array, {}, different from number of atoms {}'.format(name, len(array), len(atoms)))
            else:
                atoms.set_array(name, np.array(array))
        return atoms
    for line in fileobj.readlines():
        if line.startswith('CRYST1'):
            cellpar = [float(line[6:15]), float(line[15:24]), float(line[24:33]), float(line[33:40]), float(line[40:47]), float(line[47:54])]
            cell = cellpar_to_cell(cellpar)
            pbc = True
        for c in range(3):
            if line.startswith('ORIGX' + '123'[c]):
                orig[c] = [float(line[10:20]), float(line[20:30]), float(line[30:40])]
                trans[c] = float(line[45:55])
        if line.startswith('ATOM') or line.startswith('HETATM'):
            line_info = read_atom_line(line)
            try:
                symbol = label_to_symbol(line_info[0])
            except (KeyError, IndexError):
                symbol = label_to_symbol(line_info[1])
            position = np.dot(orig, line_info[4]) + trans
            atomtypes.append(line_info[1])
            residuenames.append(line_info[3])
            if line_info[5] is not None:
                occ.append(line_info[5])
            bfactor.append(line_info[6])
            residuenumbers.append(line_info[7])
            symbols.append(symbol)
            positions.append(position)
        if line.startswith('END'):
            atoms = build_atoms()
            images.append(atoms)
            occ = []
            bfactor = []
            residuenames = []
            atomtypes = []
            symbols = []
            positions = []
            cell = None
            pbc = None
    if len(images) == 0:
        atoms = build_atoms()
        images.append(atoms)
    return images[index]