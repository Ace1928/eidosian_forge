import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_scfout_file(filename=None):
    """
    Read the Developer output '.scfout' files. It Behaves like read_scfout.c,
    OpenMX module, but written in python. Note that some array are begin with
    1, not 0

    atomnum: the number of total atoms
    Catomnum: the number of atoms in the central region
    Latomnum: the number of atoms in the left lead
    Ratomnum: the number of atoms in the left lead
    SpinP_switch:
                 0: non-spin polarized
                 1: spin polarized
    TCpyCell: the total number of periodic cells
    Solver: method for solving eigenvalue problem
    ChemP: chemical potential
    Valence_Electrons: total number of valence electrons
    Total_SpinS: total value of Spin (2*Total_SpinS = muB)
    E_Temp: electronic temperature
    Total_NumOrbs: the number of atomic orbitals in each atom
    size: Total_NumOrbs[atomnum+1]
    FNAN: the number of first neighboring atoms of each atom
    size: FNAN[atomnum+1]
    natn: global index of neighboring atoms of an atom ct_AN
    size: natn[atomnum+1][FNAN[ct_AN]+1]
    ncn: global index for cell of neighboring atoms of an atom ct_AN
    size: ncn[atomnum+1][FNAN[ct_AN]+1]
    atv: x,y,and z-components of translation vector of periodically copied cell
    size: atv[TCpyCell+1][4]:
    atv_ijk: i,j,and j number of periodically copied cells
    size: atv_ijk[TCpyCell+1][4]:
    tv[4][4]: unit cell vectors in Bohr
    rtv[4][4]: reciprocal unit cell vectors in Bohr^{-1}
         note:
         tv_i dot rtv_j = 2PI * Kronecker's delta_{ij}
         Gxyz[atomnum+1][60]: atomic coordinates in Bohr
         Hks: Kohn-Sham matrix elements of basis orbitals
    size: Hks[SpinP_switch+1]
             [atomnum+1]
             [FNAN[ct_AN]+1]
             [Total_NumOrbs[ct_AN]]
             [Total_NumOrbs[h_AN]]
    iHks:
         imaginary Kohn-Sham matrix elements of basis orbitals
         for alpha-alpha, beta-beta, and alpha-beta spin matrices
         of which contributions come from spin-orbit coupling
         and Hubbard U effective potential.
    size: iHks[3]
              [atomnum+1]
              [FNAN[ct_AN]+1]
              [Total_NumOrbs[ct_AN]]
              [Total_NumOrbs[h_AN]]
    OLP: overlap matrix
    size: OLP[atomnum+1]
             [FNAN[ct_AN]+1]
             [Total_NumOrbs[ct_AN]]
             [Total_NumOrbs[h_AN]]
    OLPpox: overlap matrix with position operator x
    size: OLPpox[atomnum+1]
                [FNAN[ct_AN]+1]
                [Total_NumOrbs[ct_AN]]
                [Total_NumOrbs[h_AN]]
    OLPpoy: overlap matrix with position operator y
    size: OLPpoy[atomnum+1]
                [FNAN[ct_AN]+1]
                [Total_NumOrbs[ct_AN]]
                [Total_NumOrbs[h_AN]]
    OLPpoz: overlap matrix with position operator z
    size: OLPpoz[atomnum+1]
                [FNAN[ct_AN]+1]
                [Total_NumOrbs[ct_AN]]
                [Total_NumOrbs[h_AN]]
    DM: overlap matrix
    size: DM[SpinP_switch+1]
            [atomnum+1]
            [FNAN[ct_AN]+1]
            [Total_NumOrbs[ct_AN]]
            [Total_NumOrbs[h_AN]]
    dipole_moment_core[4]:
    dipole_moment_background[4]:
    """
    from numpy import insert as ins
    from numpy import cumsum as cum
    from numpy import split as spl
    from numpy import sum, zeros
    if not os.path.isfile(filename):
        return {}

    def easyReader(byte, data_type, shape):
        data_size = {'d': 8, 'i': 4}
        data_struct = {'d': float, 'i': int}
        dt = data_type
        ds = data_size[data_type]
        unpack = struct.unpack
        if len(byte) == ds:
            if dt == 'i':
                return data_struct[dt].from_bytes(byte, byteorder='little')
            elif dt == 'd':
                return np.array(unpack(dt * (len(byte) // ds), byte))[0]
        elif shape is not None:
            return np.array(unpack(dt * (len(byte) // ds), byte)).reshape(shape)
        else:
            return np.array(unpack(dt * (len(byte) // ds), byte))

    def inte(byte, shape=None):
        return easyReader(byte, 'i', shape)

    def floa(byte, shape=None):
        return easyReader(byte, 'd', shape)

    def readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f):
        myOLP = []
        myOLP.append([])
        for ct_AN in range(1, atomnum + 1):
            myOLP.append([])
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN] + 1):
                myOLP[ct_AN].append([])
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    myOLP[ct_AN][h_AN].append(floa(f.read(8 * TNO2)))
        return myOLP

    def readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, f):
        Hks = []
        for spin in range(SpinP_switch + 1):
            Hks.append([])
            Hks[spin].append([np.zeros(FNAN[0] + 1)])
            for ct_AN in range(1, atomnum + 1):
                Hks[spin].append([])
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN] + 1):
                    Hks[spin][ct_AN].append([])
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        Hks[spin][ct_AN][h_AN].append(floa(f.read(8 * TNO2)))
        return Hks
    fd = open(filename, mode='rb')
    atomnum, SpinP_switch = inte(fd.read(8))
    Catomnum, Latomnum, Ratomnum, TCpyCell = inte(fd.read(16))
    atv = floa(fd.read(8 * 4 * (TCpyCell + 1)), shape=(TCpyCell + 1, 4))
    atv_ijk = inte(fd.read(4 * 4 * (TCpyCell + 1)), shape=(TCpyCell + 1, 4))
    Total_NumOrbs = np.insert(inte(fd.read(4 * atomnum)), 0, 1, axis=0)
    FNAN = np.insert(inte(fd.read(4 * atomnum)), 0, 0, axis=0)
    natn = ins(spl(inte(fd.read(4 * sum(FNAN[1:] + 1))), cum(FNAN[1:] + 1)), 0, zeros(FNAN[0] + 1), axis=0)[:-1]
    ncn = ins(spl(inte(fd.read(4 * np.sum(FNAN[1:] + 1))), cum(FNAN[1:] + 1)), 0, np.zeros(FNAN[0] + 1), axis=0)[:-1]
    tv = ins(floa(fd.read(8 * 3 * 4), shape=(3, 4)), 0, [0, 0, 0, 0], axis=0)
    rtv = ins(floa(fd.read(8 * 3 * 4), shape=(3, 4)), 0, [0, 0, 0, 0], axis=0)
    Gxyz = ins(floa(fd.read(8 * atomnum * 4), shape=(atomnum, 4)), 0, [0.0, 0.0, 0.0, 0.0], axis=0)
    Hks = readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, fd)
    iHks = []
    if SpinP_switch == 3:
        iHks = readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, fd)
    OLP = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, fd)
    OLPpox = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, fd)
    OLPpoy = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, fd)
    OLPpoz = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, fd)
    DM = readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, fd)
    Solver = inte(fd.read(4))
    ChemP, E_Temp = floa(fd.read(8 * 2))
    dipole_moment_core = floa(fd.read(8 * 3))
    dipole_moment_background = floa(fd.read(8 * 3))
    Valence_Electrons, Total_SpinS = floa(fd.read(8 * 2))
    fd.close()
    scf_out = {'atomnum': atomnum, 'SpinP_switch': SpinP_switch, 'Catomnum': Catomnum, 'Latomnum': Latomnum, 'Hks': Hks, 'Ratomnum': Ratomnum, 'TCpyCell': TCpyCell, 'atv': atv, 'Total_NumOrbs': Total_NumOrbs, 'FNAN': FNAN, 'natn': natn, 'ncn': ncn, 'tv': tv, 'rtv': rtv, 'Gxyz': Gxyz, 'OLP': OLP, 'OLPpox': OLPpox, 'OLPpoy': OLPpoy, 'OLPpoz': OLPpoz, 'Solver': Solver, 'ChemP': ChemP, 'E_Temp': E_Temp, 'dipole_moment_core': dipole_moment_core, 'iHks': iHks, 'dipole_moment_background': dipole_moment_background, 'Valence_Electrons': Valence_Electrons, 'atv_ijk': atv_ijk, 'Total_SpinS': Total_SpinS, 'DM': DM}
    return scf_out