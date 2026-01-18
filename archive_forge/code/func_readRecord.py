import numpy
def readRecord(self):
    """Read a single fortran record"""
    L = self._read_check()
    data_str = self._read_exactly(L)
    check_size = self._read_check()
    if check_size != L:
        raise IOError('Error reading record from data file')
    return data_str