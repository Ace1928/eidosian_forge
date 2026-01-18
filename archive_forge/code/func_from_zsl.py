from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
@classmethod
def from_zsl(cls, match: ZSLMatch, film: Structure, film_miller, substrate_miller, elasticity_tensor=None, ground_state_energy=0) -> Self:
    """Generate a substrate match from a ZSL match plus metadata."""
    struct = SlabGenerator(film, film_miller, 20, 15, primitive=False).get_slab().oriented_unit_cell
    dfm = Deformation(match.match_transformation)
    strain = dfm.green_lagrange_strain.convert_to_ieee(struct, initial_fit=False)
    von_mises_strain = strain.von_mises_strain
    if elasticity_tensor is not None:
        energy_density = elasticity_tensor.energy_density(strain)
        elastic_energy = film.volume * energy_density / len(film)
    else:
        elastic_energy = 0
    return cls(film_miller=film_miller, substrate_miller=substrate_miller, strain=strain, von_mises_strain=von_mises_strain, elastic_energy=elastic_energy, ground_state_energy=ground_state_energy, **{k: getattr(match, k) for k in ['film_sl_vectors', 'substrate_sl_vectors', 'film_vectors', 'substrate_vectors', 'film_transformation', 'substrate_transformation']})