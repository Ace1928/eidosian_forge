import os.path
import io
from scipy.io.matlab._mio5 import MatFile5Reader
def read_minimat_vars(rdr):
    rdr.initialize_read()
    mdict = {'__globals__': []}
    i = 0
    while not rdr.end_of_stream():
        hdr, next_position = rdr.read_var_header()
        name = 'None' if hdr.name is None else hdr.name.decode('latin1')
        if name == '':
            name = 'var_%d' % i
            i += 1
        res = rdr.read_var_array(hdr, process=False)
        rdr.mat_stream.seek(next_position)
        mdict[name] = res
        if hdr.is_global:
            mdict['__globals__'].append(name)
    return mdict