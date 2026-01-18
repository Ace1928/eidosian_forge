import numpy as np
import pytest
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.constraints import FixAtoms
from ase.build import fcc111
from ase.io import write
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
from ase.ga.standardmutations import PermutationMutation
from ase.ga import set_raw_score
@pytest.mark.slow
def test_basic_example_main_run(seed, testdir):
    rng = np.random.RandomState(seed)
    slab = fcc111('Au', size=(4, 4, 1), vacuum=10.0, orthogonal=True)
    slab.set_constraint(FixAtoms(mask=len(slab) * [True]))
    pos = slab.get_positions()
    cell = slab.get_cell()
    p0 = np.array([0.0, 0.0, max(pos[:, 2]) + 2.0])
    v1 = cell[0, :] * 0.8
    v2 = cell[1, :] * 0.8
    v3 = cell[2, :]
    v3[2] = 3.0
    atom_numbers = 2 * [47] + 2 * [79]
    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types, ratio_of_covalent_radii=0.7)
    sg = StartGenerator(slab=slab, blocks=atom_numbers, blmin=blmin, box_to_place_in=[p0, [v1, v2, v3]], rng=rng)
    population_size = 5
    starting_population = [sg.get_new_candidate() for i in range(population_size)]
    d = PrepareDB(db_file_name=db_file, simulation_cell=slab, stoichiometry=atom_numbers)
    for a in starting_population:
        d.add_unrelaxed_candidate(a)
    population_size = 5
    mutation_probability = 0.3
    n_to_test = 5
    da = DataConnection('gadb.db')
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)
    comp = InteratomicDistanceComparator(n_top=n_to_optimize, pair_cor_cum_diff=0.015, pair_cor_max=0.7, dE=0.02, mic=False)
    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin, rng=rng)
    mutations = OperationSelector([1.0, 1.0, 1.0], [MirrorMutation(blmin, n_to_optimize, rng=rng), RattleMutation(blmin, n_to_optimize, rng=rng), PermutationMutation(n_to_optimize, rng=rng)], rng=rng)
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        a.calc = EMT()
        print('Relaxing starting candidate {0}'.format(a.info['confid']))
        with BFGS(a, trajectory=None, logfile=None) as dyn:
            dyn.run(fmax=0.05, steps=100)
        set_raw_score(a, -a.get_potential_energy())
        da.add_relaxed_step(a)
    population = Population(data_connection=da, population_size=population_size, comparator=comp, rng=rng)
    for i in range(n_to_test):
        print('Now starting configuration number {0}'.format(i))
        a1, a2 = population.get_two_candidates()
        a3, desc = pairing.get_new_individual([a1, a2])
        if a3 is None:
            continue
        da.add_unrelaxed_candidate(a3, description=desc)
        if rng.rand() < mutation_probability:
            a3_mut, desc = mutations.get_new_individual([a3])
            if a3_mut is not None:
                da.add_unrelaxed_step(a3_mut, desc)
                a3 = a3_mut
        a3.calc = EMT()
        with BFGS(a3, trajectory=None, logfile=None) as dyn:
            dyn.run(fmax=0.05, steps=100)
        set_raw_score(a3, -a3.get_potential_energy())
        da.add_relaxed_step(a3)
        population.update()
    write('all_candidates.traj', da.get_all_relaxed_candidates())