from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
@staticmethod
def buckingham_potential(structure, val_dict=None):
    """Generate species, buckingham, and spring options for an oxide structure
        using the parameters in default libraries.

        Ref:
            1. G.V. Lewis and C.R.A. Catlow, J. Phys. C: Solid State Phys.,
               18, 1149-1161 (1985)
            2. T.S.Bush, J.D.Gale, C.R.A.Catlow and P.D. Battle,
               J. Mater Chem., 4, 831-837 (1994)

        Args:
            structure: pymatgen Structure
            val_dict (Needed if structure is not charge neutral): {El:valence}
                dict, where El is element.
        """
    if not val_dict:
        try:
            el = [site.specie.symbol for site in structure]
            valences = [site.specie.oxi_state for site in structure]
            val_dict = dict(zip(el, valences))
        except AttributeError:
            bv = BVAnalyzer()
            el = [site.specie.symbol for site in structure]
            valences = bv.get_valences(structure)
            val_dict = dict(zip(el, valences))
    bpb = BuckinghamPotential('bush')
    bpl = BuckinghamPotential('lewis')
    gin = ''
    for key in val_dict:
        use_bush = True
        el = re.sub('[1-9,+,\\-]', '', key)
        if el not in bpb.species_dict or val_dict[key] != bpb.species_dict[el]['oxi']:
            use_bush = False
        if use_bush:
            gin += 'species \n'
            gin += bpb.species_dict[el]['inp_str']
            gin += 'buckingham \n'
            gin += bpb.pot_dict[el]
            gin += 'spring \n'
            gin += bpb.spring_dict[el]
            continue
        if el != 'O':
            k = f'{el}_{int(val_dict[key])}+'
            if k not in bpl.species_dict:
                raise GulpError(f'Element {k} not in library')
            gin += 'species\n'
            gin += bpl.species_dict[k]
            gin += 'buckingham\n'
            gin += bpl.pot_dict[k]
        else:
            gin += 'species\n'
            k = 'O_core'
            gin += bpl.species_dict[k]
            k = 'O_shel'
            gin += bpl.species_dict[k]
            gin += 'buckingham\n'
            gin += bpl.pot_dict[key]
            gin += 'spring\n'
            gin += bpl.spring_dict[key]
    return gin