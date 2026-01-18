import numpy as np
from ase.io.fortranfile import FortranFile
def readHSX(fname):
    """
    Read unformatted siesta HSX file
    """
    import collections
    HSX_tuple = collections.namedtuple('HSX', ['norbitals', 'norbitals_sc', 'nspin', 'nonzero', 'is_gamma', 'sc_orb2uc_orb', 'row2nnzero', 'sparse_ind2column', 'H_sparse', 'S_sparse', 'aB2RaB_sparse', 'total_elec_charge', 'temp'])
    fh = FortranFile(fname)
    norbitals, norbitals_sc, nspin, nonzero = fh.readInts('i')
    is_gamma = fh.readInts('i')[0]
    sc_orb2uc_orb = 0
    if is_gamma == 0:
        sc_orb2uc_orb = fh.readInts('i')
    row2nnzero = fh.readInts('i')
    sum_row2nnzero = np.sum(row2nnzero)
    if sum_row2nnzero != nonzero:
        raise ValueError('sum_row2nnzero != nonzero: {0} != {1}'.format(sum_row2nnzero, nonzero))
    row2displ = np.zeros(norbitals, dtype=int)
    for i in range(1, norbitals):
        row2displ[i] = row2displ[i - 1] + row2nnzero[i - 1]
    max_nonzero = np.max(row2nnzero)
    int_buff = np.zeros(max_nonzero, dtype=int)
    sparse_ind2column = np.zeros(nonzero)
    for irow in range(norbitals):
        f = row2nnzero[irow]
        int_buff[0:f] = fh.readInts('i')
        d = row2displ[irow]
        sparse_ind2column[d:d + f] = int_buff[0:f]
    sp_buff = np.zeros(max_nonzero, dtype=float)
    H_sparse = np.zeros((nonzero, nspin), dtype=float)
    S_sparse = np.zeros(nonzero, dtype=float)
    aB2RaB_sparse = np.zeros((3, nonzero), dtype=float)
    for ispin in range(nspin):
        for irow in range(norbitals):
            d = row2displ[irow]
            f = row2nnzero[irow]
            sp_buff[0:f] = fh.readReals('f')
            H_sparse[d:d + f, ispin] = sp_buff[0:f]
    for irow in range(norbitals):
        f = row2nnzero[irow]
        d = row2displ[irow]
        sp_buff[0:f] = fh.readReals('f')
        S_sparse[d:d + f] = sp_buff[0:f]
    total_elec_charge, temp = fh.readReals('d')
    sp_buff = np.zeros(3 * max_nonzero, dtype=float)
    for irow in range(norbitals):
        f = row2nnzero[irow]
        d = row2displ[irow]
        sp_buff[0:3 * f] = fh.readReals('f')
        aB2RaB_sparse[0, d:d + f] = sp_buff[0:f]
        aB2RaB_sparse[1, d:d + f] = sp_buff[f:2 * f]
        aB2RaB_sparse[2, d:d + f] = sp_buff[2 * f:3 * f]
    fh.close()
    return HSX_tuple(norbitals, norbitals_sc, nspin, nonzero, is_gamma, sc_orb2uc_orb, row2nnzero, sparse_ind2column, H_sparse, S_sparse, aB2RaB_sparse, total_elec_charge, temp)