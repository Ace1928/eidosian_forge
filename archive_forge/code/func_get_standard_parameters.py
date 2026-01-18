import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.io import ParseError
def get_standard_parameters(parameters):
    """
    Translate the OpenMX parameters to standard ASE parameters. For example,

        scf.XcType -> xc
        scf.maxIter -> maxiter
        scf.energycutoff -> energy_cutoff
        scf.Kgrid -> kpts
        scf.EigenvalueSolver -> eigensolver
        scf.SpinPolarization -> spinpol
        scf.criterion -> convergence
        scf.Electric.Field -> external
        scf.Mixing.Type -> mixer
        scf.system.charge -> charge

    We followed GPAW schem.
    """
    from ase.calculators.openmx import parameters as param
    from ase.units import Bohr, Ha, Ry, fs, m, s
    units = param.unit_dat_keywords
    standard_parameters = {}
    standard_units = {'eV': 1, 'Ha': Ha, 'Ry': Ry, 'Bohr': Bohr, 'fs': fs, 'K': 1, 'GV / m': 1000000000.0 / 1.6e-19 / m, 'Ha/Bohr': Ha / Bohr, 'm/s': m / s, '_amu': 1, 'Tesla': 1}
    translated_parameters = {'scf.XcType': 'xc', 'scf.maxIter': 'maxiter', 'scf.energycutoff': 'energy_cutoff', 'scf.Kgrid': 'kpts', 'scf.EigenvalueSolver': 'eigensolver', 'scf.SpinPolarization': 'spinpol', 'scf.criterion': 'convergence', 'scf.Electric.Field': 'external', 'scf.Mixing.Type': 'mixer', 'scf.system.charge': 'charge'}
    for key in parameters.keys():
        for openmx_key in translated_parameters.keys():
            if key == get_standard_key(openmx_key):
                standard_key = translated_parameters[openmx_key]
                unit = standard_units.get(units.get(openmx_key), 1)
                standard_parameters[standard_key] = parameters[key] * unit
    standard_parameters['spinpol'] = parameters.get('scf_spinpolarization')
    return standard_parameters