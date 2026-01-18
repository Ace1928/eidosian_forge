import numpy as np
import ase
from ase.data import chemical_symbols
from ase.utils import reader, writer
@reader
def read_cfg(fd):
    """Read atomic configuration from a CFG-file (native AtomEye format).
       See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """
    nat = None
    naux = 0
    aux = None
    auxstrs = None
    cell = np.zeros([3, 3])
    transform = np.eye(3)
    eta = np.zeros([3, 3])
    current_atom = 0
    current_symbol = None
    current_mass = None
    L = fd.readline()
    while L:
        L = L.strip()
        if len(L) != 0 and (not L.startswith('#')):
            if L == '.NO_VELOCITY.':
                vels = None
                naux += 3
            else:
                s = L.split('=')
                if len(s) == 2:
                    key, value = s
                    key = key.strip()
                    value = [x.strip() for x in value.split()]
                    if key == 'Number of particles':
                        nat = int(value[0])
                        spos = np.zeros([nat, 3])
                        masses = np.zeros(nat)
                        syms = [''] * nat
                        vels = np.zeros([nat, 3])
                        if naux > 0:
                            aux = np.zeros([nat, naux])
                    elif key == 'A':
                        pass
                    elif key == 'entry_count':
                        naux += int(value[0]) - 6
                        auxstrs = [''] * naux
                        if nat is not None:
                            aux = np.zeros([nat, naux])
                    elif key.startswith('H0('):
                        i, j = [int(x) for x in key[3:-1].split(',')]
                        cell[i - 1, j - 1] = float(value[0])
                    elif key.startswith('Transform('):
                        i, j = [int(x) for x in key[10:-1].split(',')]
                        transform[i - 1, j - 1] = float(value[0])
                    elif key.startswith('eta('):
                        i, j = [int(x) for x in key[4:-1].split(',')]
                        eta[i - 1, j - 1] = float(value[0])
                    elif key.startswith('auxiliary['):
                        i = int(key[10:-1])
                        auxstrs[i] = value[0]
                else:
                    s = [x.strip() for x in L.split()]
                    if len(s) == 1:
                        if L in chemical_symbols:
                            current_symbol = L
                        else:
                            current_mass = float(L)
                    elif current_symbol is None and current_mass is None:
                        masses[current_atom] = float(s[0])
                        syms[current_atom] = s[1]
                        spos[current_atom, :] = [float(x) for x in s[2:5]]
                        vels[current_atom, :] = [float(x) for x in s[5:8]]
                        current_atom += 1
                    elif current_symbol is not None and current_mass is not None:
                        masses[current_atom] = current_mass
                        syms[current_atom] = current_symbol
                        props = [float(x) for x in s]
                        spos[current_atom, :] = props[0:3]
                        off = 3
                        if vels is not None:
                            off = 6
                            vels[current_atom, :] = props[3:6]
                        aux[current_atom, :] = props[off:]
                        current_atom += 1
        L = fd.readline()
    if current_atom != nat:
        raise RuntimeError('Number of atoms reported for CFG file (={0}) and number of atoms actually read (={1}) differ.'.format(nat, current_atom))
    if np.any(eta != 0):
        raise NotImplementedError('eta != 0 not yet implemented for CFG reader.')
    cell = np.dot(cell, transform)
    if vels is None:
        a = ase.Atoms(symbols=syms, masses=masses, scaled_positions=spos, cell=cell, pbc=True)
    else:
        a = ase.Atoms(symbols=syms, masses=masses, scaled_positions=spos, momenta=masses.reshape(-1, 1) * vels, cell=cell, pbc=True)
    i = 0
    while i < naux:
        auxstr = auxstrs[i]
        if auxstr[-2:] == '_x':
            a.set_array(auxstr[:-2], aux[:, i:i + 3])
            i += 3
        else:
            a.set_array(auxstr, aux[:, i])
            i += 1
    return a