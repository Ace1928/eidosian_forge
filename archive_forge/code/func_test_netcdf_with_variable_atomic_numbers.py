import numpy as np
import pytest
import warnings
from ase import Atom, Atoms
from ase.io import read
from ase.io import NetCDFTrajectory
def test_netcdf_with_variable_atomic_numbers(netCDF4):
    nc = netCDF4.Dataset('6.nc', 'w')
    nc.createDimension('frame', None)
    nc.createDimension('atom', 2)
    nc.createDimension('spatial', 3)
    nc.createDimension('cell_spatial', 3)
    nc.createDimension('cell_angular', 3)
    nc.createVariable('atom_types', 'i', ('atom',))
    nc.createVariable('coordinates', 'f4', ('frame', 'atom', 'spatial'))
    nc.createVariable('cell_lengths', 'f4', ('frame', 'cell_spatial'))
    nc.createVariable('cell_angles', 'f4', ('frame', 'cell_angular'))
    r0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    r1 = 2 * r0
    nc.variables['atom_types'][:] = [1, 2]
    nc.variables['coordinates'][0] = r0
    nc.variables['coordinates'][1] = r1
    nc.variables['cell_lengths'][:] = 0
    nc.variables['cell_angles'][:] = 90
    nc.close()
    traj = NetCDFTrajectory('6.nc', 'r')
    assert np.allclose(traj[0].positions, r0)
    assert np.allclose(traj[1].positions, r1)
    traj.close()