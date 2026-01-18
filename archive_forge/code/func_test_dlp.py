from ase import io as aseIO
from ase.io.dlp4 import iread_dlp_history
from io import StringIO
import numpy as np
def test_dlp():
    cells = []
    cells.append(np.array([[23.01, -0.03943, 0.04612], [-0.09486, 22.98, 0.4551], [0.6568, 0.7694, 19.21]]))
    cells.append(np.array([[22.9, -0.03925, 0.04591], [-0.09443, 22.88, 0.4531], [0.6538, 0.766, 19.12]]))
    cells.append(np.array([[22.73, -0.03896, 0.04557], [-0.09374, 22.71, 0.4497], [0.649, 0.7603, 18.98]]))
    traj = aseIO.read(fd, format='dlp-history', index=slice(0, None))
    assert len(traj) == 3
    traj = aseIO.iread(fd, format='dlp-history', index=slice(0, None))
    for i, frame in enumerate(traj):
        assert len(frame) == 2
        assert all(frame.symbols == 'ONi')
        assert np.isclose(frame.get_cell(), cells[i]).all()
    symbols = frame.get_chemical_symbols()
    traj = iread_dlp_history(fd, symbols)
    for i, frame in enumerate(traj):
        assert len(frame) == 2
        assert all(frame.symbols == 'ONi')
        assert np.isclose(frame.get_cell(), cells[i]).all()