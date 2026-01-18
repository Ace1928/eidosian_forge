import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_kpath(self, kpts=None, symbols=None, band_kpath=None, eps=1e-05):
    """
    Convert band_kpath <-> kpts. Symbols will be guess automatically
    by using dft space group method
    For example,
    kpts  = [(0, 0, 0), (0.125, 0, 0) ... (0.875, 0, 0),
             (1, 0, 0), (1, 0.0625, 0) .. (1, 0.4375,0),
             (1, 0.5,0),(0.9375, 0.5,0).. (    ...    ),
             (0.5, 0.5, 0.5) ...               ...     ,
                ...          ...               ...     ,
                ...        (0.875, 0, 0),(1.0, 0.0, 0.0)]
    band_kpath =
    [['15','0.0','0.0','0.0','1.0','0.0','0.0','g','X'],
     ['15','1.0','0.0','0.0','1.0','0.5','0.0','X','W'],
     ['15','1.0','0.5','0.0','0.5','0.5','0.5','W','L'],
     ['15','0.5','0.5','0.5','0.0','0.0','0.0','L','g'],
     ['15','0.0','0.0','0.0','1.0','0.0','0.0','g','X']]
    where, it will be written as
     <Band.kpath
      15  0.0 0.0 0.0   1.0 0.0 0.0   g X
      15  1.0 0.0 0.0   1.0 0.5 0.0   X W
      15  1.0 0.5 0.0   0.5 0.5 0.5   W L
      15  0.5 0.5 0.5   0.0 0.0 0.0   L g
      15  0.0 0.0 0.0   1.0 0.0 0.0   g X
     Band.kpath>
    """
    if kpts is None:
        kx_linspace = np.linspace(band_kpath[0]['start_point'][0], band_kpath[0]['end_point'][0], band_kpath[0][0])
        ky_linspace = np.linspace(band_kpath[0]['start_point'][1], band_kpath[0]['end_point'][1], band_kpath[0]['kpts'])
        kz_linspace = np.linspace(band_kpath[0]['start_point'][2], band_kpath[0]['end_point'][2], band_kpath[0]['kpts'])
        kpts = np.array([kx_linspace, ky_linspace, kz_linspace]).T
        for path in band_kpath[1:]:
            kx_linspace = np.linspace(path['start_point'][0], path['end_point'][0], path['kpts'])
            ky_linspace = np.linspace(path['start_point'][1], path['end_point'][1], path['kpts'])
            kz_linspace = np.linspace(path['start_point'][2], path['end_point'][2], path['kpts'])
            k_lin = np.array([kx_linspace, ky_linspace, kz_linspace]).T
            kpts = np.append(kpts, k_lin, axis=0)
        return kpts
    elif band_kpath is None:
        band_kpath = []
        points = np.asarray(kpts)
        diffs = points[1:] - points[:-1]
        kinks = abs(diffs[1:] - diffs[:-1]).sum(1) > eps
        N = len(points)
        indices = [0]
        indices.extend(np.arange(1, N - 1)[kinks])
        indices.append(N - 1)
        for start, end, s_sym, e_sym in zip(indices[1:], indices[:-1], symbols[1:], symbols[:-1]):
            band_kpath.append({'start_point': start, 'end_point': end, 'kpts': 20, 'path_symbols': (s_sym, e_sym)})
    else:
        raise KeyError('You should specify band_kpath or kpts')
        return band_kpath