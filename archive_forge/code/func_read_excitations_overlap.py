import sys
import numpy as np
import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase
def read_excitations_overlap(self):
    """Read all finite difference excitations and wf overlaps.

        We assume that the wave function overlaps are determined as

        ov_ij = int dr displaced*_i(r) eqilibrium_j(r)
        """
    ex0 = self._eq_disp().read_exobj()
    eu = ex0.energy_to_eV_scale
    rep0_p = np.ones(len(ex0), dtype=float)

    def load(disp, rep0_p):
        ex_p = disp.read_exobj()
        ov_nn = disp.load_ov_nn()
        ov_nn = np.where(np.abs(ov_nn) > self.minoverlap['orbitals'], ov_nn, 0)
        ov_pp = ex_p.overlap(ov_nn, ex0)
        ov_pp = np.where(np.abs(ov_pp) > self.minoverlap['excitations'], ov_pp, 0)
        rep0_p *= (ov_pp.real ** 2 + ov_pp.imag ** 2).sum(axis=0)
        return (ex_p, ov_pp)

    def rotate(ex_p, ov_pp):
        e_p = np.array([ex.energy for ex in ex_p])
        m_pc = np.array([ex.get_dipole_me(form=self.dipole_form) for ex in ex_p])
        r_pp = ov_pp.T
        return ((r_pp.real ** 2 + r_pp.imag ** 2).dot(e_p), r_pp.dot(m_pc))
    exmE_rp = []
    expE_rp = []
    exF_rp = []
    exmm_rpc = []
    expm_rpc = []
    exdmdr_rpc = []
    for a, i in zip(self.myindices, self.myxyz):
        mdisp = self._disp(a, i, -1)
        pdisp = self._disp(a, i, 1)
        ex, ov = load(mdisp, rep0_p)
        exmE_p, exmm_pc = rotate(ex, ov)
        ex, ov = load(pdisp, rep0_p)
        expE_p, expm_pc = rotate(ex, ov)
        exmE_rp.append(exmE_p)
        expE_rp.append(expE_p)
        exF_rp.append(exmE_p - expE_p)
        exmm_rpc.append(exmm_pc)
        expm_rpc.append(expm_pc)
        exdmdr_rpc.append(expm_pc - exmm_pc)
    self.comm.product(rep0_p)
    select = np.where(rep0_p > self.minrep)[0]
    self.ex0E_p = np.array([ex.energy * eu for ex in ex0])[select]
    self.ex0m_pc = np.array([ex.get_dipole_me(form=self.dipole_form) for ex in ex0])[select] * u.Bohr
    if len(self.myr):
        self.exmE_rp = np.array(exmE_rp)[:, select] * eu
        self.expE_rp = np.array(expE_rp)[:, select] * eu
        self.exF_rp = np.array(exF_rp)[:, select] * eu / 2 / self.delta
        self.exmm_rpc = np.array(exmm_rpc)[:, select, :] * u.Bohr
        self.expm_rpc = np.array(expm_rpc)[:, select, :] * u.Bohr
        self.exdmdr_rpc = np.array(exdmdr_rpc)[:, select, :] * u.Bohr / 2 / self.delta
    else:
        self.exmE_rp = self.expE_rp = self.exF_rp = np.empty(0)
        self.exmm_rpc = self.expm_rpc = self.exdmdr_rpc = np.empty(0)