import os
from numpy import array
import ase
import ase.io
def test_eon():
    con_file = 'neb.con'
    with open(con_file, 'w') as fd:
        fd.write(CON_FILE)
    images = ase.io.read(con_file, format='eon', index=':')
    box = images[-2]
    assert abs(box.cell - data.cell).sum() < TOL
    assert abs(box.positions - data.positions).sum() < TOL
    out_file = 'out.con'
    ase.io.write(out_file, data, format='eon')
    data2 = ase.io.read(out_file, format='eon')
    assert abs(data2.cell - data.cell).sum() < TOL
    assert abs(data2.positions - data.positions).sum() < TOL