import numpy as np
from ase.io.fortranfile import FortranFile
def readWFSX(fname):
    """
    Read unformatted siesta WFSX file
    """
    import collections
    import struct
    WFSX_tuple = collections.namedtuple('WFSX', ['nkpoints', 'nspin', 'norbitals', 'gamma', 'orb2atm', 'orb2strspecies', 'orb2ao', 'orb2n', 'orb2strsym', 'kpoints', 'DFT_E', 'DFT_X', 'mo_spin_kpoint_2_is_read'])
    fh = FortranFile(fname)
    nkpoints, gamma = fh.readInts('i')
    nspin = fh.readInts('i')[0]
    norbitals = fh.readInts('i')[0]
    orb2atm = np.zeros(norbitals, dtype=int)
    orb2strspecies = []
    orb2ao = np.zeros(norbitals, dtype=int)
    orb2n = np.zeros(norbitals, dtype=int)
    orb2strsym = []
    dat_size = struct.calcsize('i20sii20s')
    dat = fh.readRecord()
    ind_st = 0
    ind_fn = dat_size
    for iorb in range(norbitals):
        val_list = struct.unpack('i20sii20s', dat[ind_st:ind_fn])
        orb2atm[iorb] = val_list[0]
        orb2strspecies.append(val_list[1])
        orb2ao[iorb] = val_list[2]
        orb2n[iorb] = val_list[3]
        orb2strsym.append(val_list[4])
        ind_st = ind_st + dat_size
        ind_fn = ind_fn + dat_size
    orb2strspecies = np.array(orb2strspecies)
    orb2strsym = np.array(orb2strsym)
    kpoints = np.zeros((3, nkpoints), dtype=np.float64)
    DFT_E = np.zeros((norbitals, nspin, nkpoints), dtype=np.float64)
    if gamma == 1:
        DFT_X = np.zeros((1, norbitals, norbitals, nspin, nkpoints), dtype=np.float64)
        eigenvector = np.zeros((1, norbitals), dtype=float)
    else:
        DFT_X = np.zeros((2, norbitals, norbitals, nspin, nkpoints), dtype=np.float64)
        eigenvector = np.zeros((2, norbitals), dtype=float)
    mo_spin_kpoint_2_is_read = np.zeros((norbitals, nspin, nkpoints), dtype=bool)
    mo_spin_kpoint_2_is_read[0:norbitals, 0:nspin, 0:nkpoints] = False
    dat_size = struct.calcsize('iddd')
    for ikpoint in range(nkpoints):
        for ispin in range(nspin):
            dat = fh.readRecord()
            val_list = struct.unpack('iddd', dat[0:dat_size])
            ikpoint_in = val_list[0] - 1
            kpoints[0:3, ikpoint] = val_list[1:4]
            if ikpoint != ikpoint_in:
                raise ValueError('siesta_get_wfsx: ikpoint != ikpoint_in')
            ispin_in = fh.readInts('i')[0] - 1
            if ispin_in > nspin - 1:
                msg = 'siesta_get_wfsx: err: ispin_in>nspin\n                      siesta_get_wfsx: ikpoint, ispin, ispin_in =                      {0}  {1}  {2}\n siesta_get_wfsx'.format(ikpoint, ispin, ispin_in)
                raise ValueError(msg)
            norbitals_in = fh.readInts('i')[0]
            if norbitals_in > norbitals:
                msg = 'siesta_get_wfsx: err: norbitals_in>norbitals\n                      siesta_get_wfsx: ikpoint, norbitals, norbitals_in =                      {0}  {1}  {2}\n siesta_get_wfsx'.format(ikpoint, norbitals, norbitals_in)
                raise ValueError(msg)
            for imolecular_orb in range(norbitals_in):
                imolecular_orb_in = fh.readInts('i')[0] - 1
                if imolecular_orb_in > norbitals - 1:
                    msg = '\n                        siesta_get_wfsx: err: imolecular_orb_in>norbitals\n\n                        siesta_get_wfsx: ikpoint, norbitals,\n                        imolecular_orb_in = {0}  {1}  {2}\n\n                        siesta_get_wfsx'.format(ikpoint, norbitals, imolecular_orb_in)
                    raise ValueError(msg)
                real_E_eV = fh.readReals('d')[0]
                eigenvector = fh.readReals('f')
                DFT_E[imolecular_orb_in, ispin_in, ikpoint] = real_E_eV / 13.6058
                DFT_X[:, :, imolecular_orb_in, ispin_in, ikpoint] = eigenvector
                mo_spin_kpoint_2_is_read[imolecular_orb_in, ispin_in, ikpoint] = True
            if not all(mo_spin_kpoint_2_is_read[:, ispin_in, ikpoint]):
                msg = 'siesta_get_wfsx: warn: .not. all(mo_spin_k_2_is_read)'
                print('mo_spin_kpoint_2_is_read = ', mo_spin_kpoint_2_is_read)
                raise ValueError(msg)
    fh.close()
    return WFSX_tuple(nkpoints, nspin, norbitals, gamma, orb2atm, orb2strspecies, orb2ao, orb2n, orb2strsym, kpoints, DFT_E, DFT_X, mo_spin_kpoint_2_is_read)