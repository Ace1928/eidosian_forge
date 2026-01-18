import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.calculators.calculator import PropertyNotImplementedError
def calculate_band_structure(atoms, path=None, scf_kwargs=None, bs_kwargs=None, kpts_tol=1e-06, cell_tol=1e-06):
    """Calculate band structure.

    The purpose of this function is to abstract a band structure calculation
    so the workflow does not depend on the calculator.

    First trigger SCF calculation if necessary, then set arguments
    on the calculator for band structure calculation, then return
    calculated band structure.

    The difference from get_band_structure() is that the latter
    expects the calculation to already have been done."""
    if path is None:
        path = atoms.cell.bandpath()
    from ase.lattice import celldiff
    if any(path.cell.any(1) != atoms.pbc):
        raise ValueError("The band path's cell, {}, does not match the periodicity {} of the atoms".format(path.cell, atoms.pbc))
    cell_err = celldiff(path.cell, atoms.cell.uncomplete(atoms.pbc))
    if cell_err > cell_tol:
        raise ValueError('Atoms and band path have different unit cells.  Please reduce atoms to standard form.  Cell lengths and angles are {} vs {}'.format(atoms.cell.cellpar(), path.cell.cellpar()))
    calc = atoms.calc
    if calc is None:
        raise ValueError('Atoms have no calculator')
    if scf_kwargs is not None:
        calc.set(**scf_kwargs)
    use_bandpath_kw = getattr(calc, 'accepts_bandpath_keyword', False)
    if use_bandpath_kw:
        calc.set(bandpath=path)
        atoms.get_potential_energy()
        return calc.band_structure()
    atoms.get_potential_energy()
    if hasattr(calc, 'get_fermi_level'):
        eref = calc.get_fermi_level()
    else:
        eref = 0.0
    if bs_kwargs is None:
        bs_kwargs = {}
    calc.set(kpts=path, **bs_kwargs)
    calc.results.clear()
    try:
        atoms.get_potential_energy()
    except PropertyNotImplementedError:
        pass
    ibzkpts = calc.get_ibz_k_points()
    kpts_err = np.abs(path.kpts - ibzkpts).max()
    if kpts_err > kpts_tol:
        raise RuntimeError('Kpoints of calculator differ from those of the band path we just used; err={} > tol={}'.format(kpts_err, kpts_tol))
    bs = get_band_structure(atoms, path=path, reference=eref)
    return bs