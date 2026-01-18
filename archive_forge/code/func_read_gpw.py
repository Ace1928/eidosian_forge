from ase import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.units import Bohr, Hartree
import ase.io.ulm as ulm
from ase.io.trajectory import read_atoms
def read_gpw(filename):
    try:
        reader = ulm.open(filename)
    except ulm.InvalidULMFileError:
        return read_old_gpw(filename)
    atoms = read_atoms(reader.atoms, _try_except=False)
    wfs = reader.wave_functions
    kpts = wfs.get('kpts')
    if kpts is None:
        ibzkpts = None
        bzkpts = None
        bz2ibz = None
    else:
        ibzkpts = kpts.ibzkpts
        bzkpts = kpts.get('bzkpts')
        bz2ibz = kpts.get('bz2ibz')
    if reader.version >= 3:
        efermi = reader.wave_functions.fermi_levels.mean()
    else:
        efermi = reader.occupations.fermilevel
    atoms.calc = SinglePointDFTCalculator(atoms, efermi=efermi, ibzkpts=ibzkpts, bzkpts=bzkpts, bz2ibz=bz2ibz, **reader.results.asdict())
    if kpts is not None:
        atoms.calc.kpts = []
        spin = 0
        for eps_kn, f_kn in zip(wfs.eigenvalues, wfs.occupations):
            kpt = 0
            for weight, eps_n, f_n in zip(kpts.weights, eps_kn, f_kn):
                atoms.calc.kpts.append(SinglePointKPoint(weight, spin, kpt, eps_n, f_n))
                kpt += 1
            spin += 1
    return atoms