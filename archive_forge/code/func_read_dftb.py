import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
@reader
def read_dftb(fd):
    """Method to read coordinates from the Geometry section
    of a DFTB+ input file (typically called "dftb_in.hsd").

    As described in the DFTB+ manual, this section can be
    in a number of different formats. This reader supports
    the GEN format and the so-called "explicit" format.

    The "explicit" format is unique to DFTB+ input files.
    The GEN format can also be used in a stand-alone fashion,
    as coordinate files with a `.gen` extension. Reading and
    writing such files is implemented in `ase.io.gen`.
    """
    lines = fd.readlines()
    atoms_pos = []
    atom_symbols = []
    type_names = []
    my_pbc = False
    fractional = False
    mycell = []
    for iline, line in enumerate(lines):
        if line.strip().startswith('#'):
            pass
        elif 'genformat' in line.lower():
            natoms = int(lines[iline + 1].split()[0])
            if lines[iline + 1].split()[1].lower() == 's':
                my_pbc = True
            elif lines[iline + 1].split()[1].lower() == 'f':
                my_pbc = True
                fractional = True
            symbols = lines[iline + 2].split()
            for i in range(natoms):
                index = iline + 3 + i
                aindex = int(lines[index].split()[1]) - 1
                atom_symbols.append(symbols[aindex])
                position = [float(p) for p in lines[index].split()[2:]]
                atoms_pos.append(position)
            if my_pbc:
                for i in range(3):
                    index = iline + 4 + natoms + i
                    cell = [float(c) for c in lines[index].split()]
                    mycell.append(cell)
        elif 'TypeNames' in line:
            col = line.split()
            for i in range(3, len(col) - 1):
                type_names.append(col[i].strip('"'))
        elif 'Periodic' in line:
            if 'Yes' in line:
                my_pbc = True
        elif 'LatticeVectors' in line:
            for imycell in range(3):
                extraline = lines[iline + imycell + 1]
                cols = extraline.split()
                mycell.append([float(cols[0]), float(cols[1]), float(cols[2])])
        else:
            pass
    if not my_pbc:
        mycell = [0.0] * 3
    start_reading_coords = False
    stop_reading_coords = False
    for line in lines:
        if line.strip().startswith('#'):
            pass
        else:
            if 'TypesAndCoordinates' in line:
                start_reading_coords = True
            if start_reading_coords:
                if '}' in line:
                    stop_reading_coords = True
            if start_reading_coords and (not stop_reading_coords) and ('TypesAndCoordinates' not in line):
                typeindexstr, xxx, yyy, zzz = line.split()[:4]
                typeindex = int(typeindexstr)
                symbol = type_names[typeindex - 1]
                atom_symbols.append(symbol)
                atoms_pos.append([float(xxx), float(yyy), float(zzz)])
    if fractional:
        atoms = Atoms(scaled_positions=atoms_pos, symbols=atom_symbols, cell=mycell, pbc=my_pbc)
    elif not fractional:
        atoms = Atoms(positions=atoms_pos, symbols=atom_symbols, cell=mycell, pbc=my_pbc)
    return atoms