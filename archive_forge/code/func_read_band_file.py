import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def read_band_file(filename=None):
    band_data = {}
    if not os.path.isfile(filename):
        return {}
    band_kpath = []
    eigen_bands = []
    with open(filename, 'r') as fd:
        line = f.readline().split()
        nkpts = 0
        nband = int(line[0])
        nspin = int(line[1]) + 1
        band_data['nband'] = nband
        band_data['nspin'] = nspin
        line = f.readline().split()
        band_data['band_kpath_unitcell'] = [line[:3], line[3:6], line[6:9]]
        line = f.readline().split()
        band_data['band_nkpath'] = int(line[0])
        for i in range(band_data['band_nkpath']):
            line = f.readline().split()
            band_kpath.append(line)
            nkpts += int(line[0])
        band_data['nkpts'] = nkpts
        band_data['band_kpath'] = band_kpath
        kpts = np.zeros((nkpts, 3))
        eigen_bands = np.zeros((nspin, nkpts, nband))
        for i in range(nspin):
            for j in range(nkpts):
                line = f.readline()
                kpts[j] = np.array(line.split(), dtype=float)[1:]
                line = f.readline()
                eigen_bands[i, j] = np.array(line.split(), dtype=float)[:]
        band_data['eigenvalues'] = eigen_bands
        band_data['band_kpts'] = kpts
    return band_data