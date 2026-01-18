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
def epsilon_imag(cder: NDArray, eigs: NDArray, kweights: ArrayLike, efermi: float, nedos: int, deltae: float, ismear: int, sigma: float, idir: int, jdir: int, mask: NDArray | None=None):
    """Replicate the EPSILON_IMAG function of VASP.

    Args:
        cder: The data written to the WAVEDER (nbands, nbands, nkpoints, nspin, diri, dirj)
        eigs: The eigenvalues (nbands, nkpoints, nspin)
        kweights: The kpoint weights (nkpoints)
        efermi: The fermi energy
        nedos: The sampling of the energy values
        deltae: The energy grid spacing
        ismear: The smearing parameter used by the ``step_func``.
        sigma: The width of the smearing
        idir: The first direction of the dielectric tensor
        jdir: The second direction of the dielectric tensor
        mask: Mask for the bands/kpoint/spin index to include in the calculation

    Returns:
        np.array: Array of size `nedos` with the imaginary part of the dielectric function.
    """
    norm_kweights = np.array(kweights) / np.sum(kweights)
    egrid = np.linspace(0, nedos * deltae, nedos, endpoint=False)
    eigs_shifted = eigs - efermi
    rspin = 3 - cder.shape[3]
    cderm = cder * mask if mask is not None else cder
    try:
        min_band0, max_band0 = (np.min(np.where(cderm)[0]), np.max(np.where(cderm)[0]))
        min_band1, max_band1 = (np.min(np.where(cderm)[1]), np.max(np.where(cderm)[1]))
    except ValueError as exc:
        if 'zero-size array' in str(exc):
            return (egrid, np.zeros_like(egrid, dtype=np.complex128))
        raise exc
    _, _, nk, nspin = cderm.shape[:4]
    iter_idx = [range(min_band0, max_band0 + 1), range(min_band1, max_band1 + 1), range(nk), range(nspin)]
    num_ = (max_band0 - min_band0) * (max_band1 - min_band1) * nk * nspin
    epsdd = np.zeros_like(egrid, dtype=np.complex128)
    for ib, jb, ik, ispin in tqdm(itertools.product(*iter_idx), total=num_):
        fermi_w_i = step_func(eigs_shifted[ib, ik, ispin] / sigma, ismear)
        fermi_w_j = step_func(eigs_shifted[jb, ik, ispin] / sigma, ismear)
        weight = (fermi_w_j - fermi_w_i) * rspin * norm_kweights[ik]
        decel = eigs[jb, ik, ispin] - eigs[ib, ik, ispin]
        A = cderm[ib, jb, ik, ispin, idir] * np.conjugate(cderm[ib, jb, ik, ispin, jdir])
        smeared = get_delta(x0=decel, sigma=sigma, nx=nedos, dx=deltae, ismear=ismear) * weight * A
        epsdd += smeared
    return (egrid, epsdd)