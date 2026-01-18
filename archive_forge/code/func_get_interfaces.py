from __future__ import annotations
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import polar
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, fast_norm
from pymatgen.core.interface import Interface, label_termination
from pymatgen.core.surface import SlabGenerator
def get_interfaces(self, termination: tuple[str, str], gap: float=2.0, vacuum_over_film: float=20.0, film_thickness: float=1, substrate_thickness: float=1, in_layers: bool=True) -> Iterator[Interface]:
    """Generates interface structures given the film and substrate structure
        as well as the desired terminations.

        Args:
            termination (tuple[str, str]): termination from self.termination list
            gap (float, optional): gap between film and substrate. Defaults to 2.0.
            vacuum_over_film (float, optional): vacuum over the top of the film. Defaults to 20.0.
            film_thickness (float, optional): the film thickness. Defaults to 1.
            substrate_thickness (float, optional): substrate thickness. Defaults to 1.
            in_layers (bool, optional): set the thickness in layer units. Defaults to True.

        Yields:
            Iterator[Interface]: interfaces from slabs
        """
    film_sg = SlabGenerator(self.film_structure, self.film_miller, min_slab_size=film_thickness, min_vacuum_size=3, in_unit_planes=in_layers, center_slab=True, primitive=True, reorient_lattice=False)
    sub_sg = SlabGenerator(self.substrate_structure, self.substrate_miller, min_slab_size=substrate_thickness, min_vacuum_size=3, in_unit_planes=in_layers, center_slab=True, primitive=True, reorient_lattice=False)
    film_shift, sub_shift = self._terminations[termination]
    film_slab = film_sg.get_slab(shift=film_shift)
    sub_slab = sub_sg.get_slab(shift=sub_shift)
    for match in self.zsl_matches:
        super_film_transform = np.round(from_2d_to_3d(get_2d_transform(film_slab.lattice.matrix[:2], match.film_sl_vectors))).astype(int)
        film_sl_slab = film_slab.copy()
        film_sl_slab.make_supercell(super_film_transform)
        assert_allclose(film_sl_slab.lattice.matrix[2], film_slab.lattice.matrix[2], atol=1e-08, err_msg='2D transformation affected C-axis for Film transformation')
        assert_allclose(film_sl_slab.lattice.matrix[:2], match.film_sl_vectors, atol=1e-08, err_msg="Transformation didn't make proper supercell for film")
        super_sub_transform = np.round(from_2d_to_3d(get_2d_transform(sub_slab.lattice.matrix[:2], match.substrate_sl_vectors))).astype(int)
        sub_sl_slab = sub_slab.copy()
        sub_sl_slab.make_supercell(super_sub_transform)
        assert_allclose(sub_sl_slab.lattice.matrix[2], sub_slab.lattice.matrix[2], atol=1e-08, err_msg='2D transformation affected C-axis for Film transformation')
        assert_allclose(sub_sl_slab.lattice.matrix[:2], match.substrate_sl_vectors, atol=1e-08, err_msg="Transformation didn't make proper supercell for substrate")
        match_dict = match.as_dict()
        interface_properties = {k: match_dict[k] for k in match_dict if not k.startswith('@')}
        dfm = Deformation(match.match_transformation)
        strain = dfm.green_lagrange_strain
        interface_properties['strain'] = strain
        interface_properties['von_mises_strain'] = strain.von_mises_strain
        interface_properties['termination'] = termination
        interface_properties['film_thickness'] = film_thickness
        interface_properties['substrate_thickness'] = substrate_thickness
        yield Interface.from_slabs(substrate_slab=sub_sl_slab, film_slab=film_sl_slab, gap=gap, vacuum_over_film=vacuum_over_film, interface_properties=interface_properties)