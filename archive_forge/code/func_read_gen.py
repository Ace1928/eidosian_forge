from ase.atoms import Atoms
from ase.utils import reader, writer
@reader
def read_gen(fileobj):
    """Read structure in GEN format (refer to DFTB+ manual).
       Multiple snapshot are not allowed. """
    image = Atoms()
    lines = fileobj.readlines()
    line = lines[0].split()
    natoms = int(line[0])
    pb_flag = line[1]
    if line[1] not in ['C', 'F', 'S']:
        raise IOError('Error in line #1: only C (Cluster), S (Supercell) or F (Fraction) are valid options')
    line = lines[1].split()
    symboldict = dict()
    symbolid = 1
    for symb in line:
        symboldict[symbolid] = symb
        symbolid += 1
    del lines[:2]
    positions = []
    symbols = []
    for line in lines[:natoms]:
        dummy, symbolid, x, y, z = line.split()[:5]
        symbols.append(symboldict[int(symbolid)])
        positions.append([float(x), float(y), float(z)])
    image = Atoms(symbols=symbols, positions=positions)
    del lines[:natoms]
    if pb_flag == 'C':
        return image
    else:
        del lines[:1]
        image.set_pbc([True, True, True])
        p = []
        for i in range(3):
            x, y, z = lines[i].split()[:3]
            p.append([float(x), float(y), float(z)])
        image.set_cell([(p[0][0], p[0][1], p[0][2]), (p[1][0], p[1][1], p[1][2]), (p[2][0], p[2][1], p[2][2])])
        if pb_flag == 'F':
            frac_positions = image.get_positions()
            image.set_scaled_positions(frac_positions)
        return image