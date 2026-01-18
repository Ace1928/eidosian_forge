import pytest
import numpy as np
from ase import Atoms
@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpslib')
def test_change_cell_dimensions_and_pbc(factory, dimer_params, lj_epsilons):
    """Ensure that post_change_box commands are actually executed after
    changing the dimensions of the cell or its periodicity.  This is done by
    setting up an isolated dimer with a Lennard-Jones potential and a set of
    post_changebox_cmds that specify the same potential but with a rescaled
    energy (epsilon) parameter.  The energy is then computed twice, once before
    changing the cell dimensions and once after, and the values are compared to
    the expected values based on the two different epsilons to ensure that the
    modified LJ potential is used for the second calculation.  The procedure is
    repeated but where the periodicity of the cell boundaries is changed rather
    than the cell dimensions.
    """
    dimer = Atoms(**dimer_params)
    spec, a = extract_dimer_species_and_separation(dimer)
    lj_cutoff = 3 * a
    calc_params = calc_params_lj_changebox(spec, lj_cutoff, **lj_epsilons)
    dimer.calc = factory.calc(**calc_params)
    energy_orig = dimer.get_potential_energy()
    cell_orig = dimer.get_cell()
    dimer.set_cell(cell_orig * 1.01, scale_atoms=False)
    energy_modified = dimer.get_potential_energy()
    eps_scaling_factor = lj_epsilons['eps_modified'] / lj_epsilons['eps_orig']
    assert energy_modified == pytest.approx(eps_scaling_factor * energy_orig, rel=0.0001)
    dimer.set_cell(cell_orig, scale_atoms=False)
    dimer.calc = factory.calc(**calc_params)
    energy_orig = dimer.get_potential_energy()
    dimer.set_pbc([False, True, False])
    energy_modified = dimer.get_potential_energy()
    assert energy_modified == pytest.approx(eps_scaling_factor * energy_orig, rel=0.0001)