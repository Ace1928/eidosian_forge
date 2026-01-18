from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants
import scipy.special
from monty.json import MSONable
from tqdm import tqdm
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import Vasprun, Waveder
@classmethod
def from_vasp_objects(cls, vrun: Vasprun, waveder: Waveder) -> Self:
    """Construct a DielectricFunction from Vasprun, Kpoint, and Waveder objects.

        Args:
            vrun: Vasprun object
            kpoint: Kpoint object
            waveder: Waveder object
        """
    bands = vrun.eigenvalues
    sspins = [Spin.up, Spin.down]
    eigs = np.stack([bands[spin] for spin in sspins[:vrun.parameters['ISPIN']]], axis=2)[..., 0]
    eigs = np.swapaxes(eigs, 0, 1)
    kweights = vrun.actual_kpoints_weights
    nedos = vrun.parameters['NEDOS']
    deltae = vrun.dielectric[0][1]
    ismear = vrun.parameters['ISMEAR']
    sigma = vrun.parameters['SIGMA']
    cshift = vrun.parameters['CSHIFT']
    efermi = vrun.efermi
    ispin = vrun.parameters['ISPIN']
    volume = vrun.final_structure.volume
    if vrun.parameters['ISYM'] != 0:
        raise NotImplementedError('ISYM != 0 is not implemented yet')
    return cls(cder_real=waveder.cder_real, cder_imag=waveder.cder_imag, eigs=eigs, kweights=kweights, nedos=nedos, deltae=deltae, ismear=ismear, sigma=sigma, efermi=efermi, cshift=cshift, ispin=ispin, volume=volume)