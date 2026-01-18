from __future__ import annotations
import re
import numpy as np
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
from pymatgen.analysis.chemenv.utils.chemenv_errors import NeighborsNotComputedChemenvError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import rotateCoords
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Molecule
from pymatgen.io.cif import CifParser
def compute_environments(chemenv_configuration):
    """
    Compute the environments.

    Args:
        chemenv_configuration:
    """
    string_sources = {'cif': {'string': 'a Cif file', 'regexp': '.*\\.cif$'}, 'mp': {'string': 'the Materials Project database', 'regexp': 'mp-[0-9]+$'}}
    questions = {'c': 'cif'}
    questions['m'] = 'mp'
    lgf = LocalGeometryFinder()
    lgf.setup_parameters()
    all_cg = AllCoordinationGeometries()
    strategy_class = strategies_class_lookup[chemenv_configuration.package_options['default_strategy']['strategy']]
    default_strategy = strategy_class()
    default_strategy.setup_options(chemenv_configuration.package_options['default_strategy']['strategy_options'])
    max_dist_factor = chemenv_configuration.package_options['default_max_distance_factor']
    first_time = True
    while True:
        if len(questions) > 1:
            found = False
            print('Enter the source from which the structure is coming or <q> to quit :')
            for key_character, qq in questions.items():
                print(f' - <{key_character}> for a structure from {string_sources[qq]['string']}')
            test = input(' ... ')
            if test == 'q':
                break
            if test not in list(questions):
                for qq in questions.values():
                    if re.match(string_sources[qq]['regexp'], str(test)) is not None:
                        found = True
                        source_type = qq
                if not found:
                    print('Wrong key, try again ...')
                    continue
            else:
                source_type = questions[test]
        else:
            found = False
            source_type = next(iter(questions.values()))
        if found and len(questions) > 1:
            input_source = test
        if source_type == 'cif':
            if not found:
                input_source = input('Enter path to CIF file : ')
            parser = CifParser(input_source)
            structure = parser.parse_structures(primitive=True)[0]
        elif source_type == 'mp':
            if not found:
                input_source = input('Enter materials project id (e.g. "mp-1902") : ')
            from pymatgen.ext.matproj import MPRester
            with MPRester() as mpr:
                structure = mpr.get_structure_by_material_id(input_source)
        lgf.setup_structure(structure)
        print(f'Computing environments for {structure.reduced_formula} ... ')
        se = lgf.compute_structure_environments(maximum_distance_factor=max_dist_factor)
        print('Computing environments finished')
        while True:
            test = input('See list of environments determined for each (inequivalent) site ? ("y" or "n", "d" with details, "g" to see the grid) : ')
            strategy = default_strategy
            if test in ['y', 'd', 'g']:
                strategy.set_structure_environments(se)
                for equiv_list in se.equivalent_sites:
                    site = equiv_list[0]
                    site_idx = se.structure.index(site)
                    try:
                        if strategy.uniquely_determines_coordination_environments:
                            ces = strategy.get_site_coordination_environments(site)
                        else:
                            ces = strategy.get_site_coordination_environments_fractions(site)
                    except NeighborsNotComputedChemenvError:
                        continue
                    if ces is None:
                        continue
                    if len(ces) == 0:
                        continue
                    comp = site.species
                    reduced_formula = comp.get_reduced_formula_and_factor()[0]
                    if strategy.uniquely_determines_coordination_environments:
                        ce = ces[0]
                        if ce is None:
                            continue
                        the_cg = all_cg.get_geometry_from_mp_symbol(ce[0])
                        msg = f'Environment for site #{site_idx} {reduced_formula} ({comp}) : {the_cg.name} ({ce[0]})\n'
                    else:
                        msg = f'Environments for site #{site_idx} {reduced_formula} ({comp}) : \n'
                        for ce in ces:
                            cg = all_cg.get_geometry_from_mp_symbol(ce[0])
                            csm = ce[1]['other_symmetry_measures']['csm_wcs_ctwcc']
                            msg += f' - {cg.name} ({cg.mp_symbol}): {ce[2]:.2%} (csm : {csm:2f})\n'
                    if test in ['d', 'g'] and strategy.uniquely_determines_coordination_environments and (the_cg.mp_symbol != UNCLEAR_ENVIRONMENT_SYMBOL):
                        msg += '  <Continuous symmetry measures>  '
                        min_geoms = se.ce_list[site_idx][the_cg.coordination_number][0].minimum_geometries()
                        for min_geom in min_geoms:
                            csm = min_geom[1]['other_symmetry_measures']['csm_wcs_ctwcc']
                            msg += f'{min_geom[0]} : {csm:.2f}       '
                    print(msg)
            if test == 'g':
                while True:
                    test = input('Enter index of site(s) (e.g. 0 1 2, separated by spaces) for which you want to see the grid of parameters : ')
                    try:
                        indices = [int(x) for x in test.split()]
                        print(str(indices))
                        for site_idx in indices:
                            if site_idx < 0:
                                raise IndexError
                            se.plot_environments(site_idx)
                        break
                    except ValueError:
                        print('This is not a valid site')
                    except IndexError:
                        print('This site is out of the site range')
            if StructureVis is None:
                test = input('Go to next structure ? ("y" to do so)')
                if test == 'y':
                    break
                continue
            test = input('View structure with environments ? ("y" for the unit cell or "m" for a supercell or "n") : ')
            if test in ['y', 'm']:
                if test == 'm':
                    deltas = []
                    while True:
                        try:
                            test = input('Enter multiplicity (e.g. 3 2 2) : ')
                            nns = test.split()
                            for i0 in range(int(nns[0])):
                                for i1 in range(int(nns[1])):
                                    for i2 in range(int(nns[2])):
                                        deltas.append(np.array([1.0 * i0, 1.0 * i1, 1.0 * i2], float))
                            break
                        except (ValueError, IndexError):
                            print('Not a valid multiplicity')
                else:
                    deltas = [np.zeros(3, float)]
                if first_time and StructureVis is not None:
                    vis = StructureVis(show_polyhedron=False, show_unit_cell=True)
                    vis.show_help = False
                    first_time = False
                vis.set_structure(se.structure)
                strategy.set_structure_environments(se)
                for site in se.structure:
                    try:
                        ces = strategy.get_site_coordination_environments(site)
                    except NeighborsNotComputedChemenvError:
                        continue
                    if len(ces) == 0:
                        continue
                    ce = strategy.get_site_coordination_environment(site)
                    if ce is not None and ce[0] != UNCLEAR_ENVIRONMENT_SYMBOL:
                        for delta in deltas:
                            psite = PeriodicSite(site.species, site.frac_coords + delta, site.lattice, properties=site.properties)
                            vis.add_site(psite)
                            neighbors = strategy.get_site_neighbors(psite)
                            draw_cg(vis, psite, neighbors, cg=lgf.allcg.get_geometry_from_mp_symbol(ce[0]), perm=ce[1]['permutation'])
                vis.show()
            test = input('Go to next structure ? ("y" to do so) : ')
            if test == 'y':
                break
        print()