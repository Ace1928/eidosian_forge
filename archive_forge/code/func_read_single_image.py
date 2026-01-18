import re
from numpy import zeros, isscalar
from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator
def read_single_image(f, levcfg, imcon, natoms, is_trajectory, symbols=None):
    cell = zeros((3, 3))
    ispbc = imcon > 0
    if ispbc or is_trajectory:
        for j in range(3):
            line = f.readline()
            line = line.split()
            for i in range(3):
                try:
                    cell[j, i] = float(line[i])
                except ValueError:
                    raise RuntimeError('error reading cell')
    if symbols:
        sym = symbols
    else:
        sym = []
    positions = []
    velocities = []
    forces = []
    charges = []
    masses = []
    disp = []
    if is_trajectory:
        counter = range(natoms)
    else:
        from itertools import count
        counter = count()
    labels = []
    mass = None
    ch = None
    d = None
    for a in counter:
        line = f.readline()
        if not line:
            a -= 1
            break
        m = re.match('\\s*([A-Za-z][a-z]?)(\\S*)', line)
        assert m is not None, line
        symbol, label = m.group(1, 2)
        symbol = symbol.capitalize()
        if is_trajectory:
            ll = line.split()
            if len(ll) == 5:
                mass, ch, d = [float(i) for i in line.split()[2:5]]
            else:
                mass, ch = [float(i) for i in line.split()[2:4]]
            charges.append(ch)
            masses.append(mass)
            disp.append(d)
        if not symbols:
            assert symbol in chemical_symbols
            sym.append(symbol)
        if label:
            labels.append(label)
        else:
            labels.append('')
        x, y, z = f.readline().split()[:3]
        positions.append([float(x), float(y), float(z)])
        if levcfg > 0:
            vx, vy, vz = f.readline().split()[:3]
            velocities.append([float(vx) * dlp_v_ase, float(vy) * dlp_v_ase, float(vz) * dlp_v_ase])
        if levcfg > 1:
            fx, fy, fz = f.readline().split()[:3]
            forces.append([float(fx) * dlp_f_ase, float(fy) * dlp_f_ase, float(fz) * dlp_f_ase])
    if symbols and a + 1 != len(symbols):
        raise ValueError('Counter is at {} but you gave {} symbols'.format(a + 1, len(symbols)))
    if imcon == 0:
        pbc = False
    elif imcon == 6:
        pbc = [True, True, False]
    else:
        assert imcon in [1, 2, 3]
        pbc = True
    atoms = Atoms(positions=positions, symbols=sym, cell=cell, pbc=pbc, celldisp=-cell.sum(axis=0) / 2)
    if is_trajectory:
        atoms.set_masses(masses)
        atoms.set_array(DLP4_DISP_KEY, disp, float)
        atoms.set_initial_charges(charges)
    atoms.set_array(DLP4_LABELS_KEY, labels, str)
    if levcfg > 0:
        atoms.set_velocities(velocities)
    if levcfg > 1:
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
    return atoms