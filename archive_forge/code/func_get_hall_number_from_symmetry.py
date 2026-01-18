from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_hall_number_from_symmetry(rotations, translations, symprec=1e-05) -> int | None:
    """Hall number is obtained from a set of symmetry operations. If fails, return None.

    .. deprecated:: 2.0
        Replaced by {func}`get_spacegroup_type_from_symmetry`.

    Return one of ``hall_number`` corresponding to a space-group type of the given
    set of symmetry operations. When multiple ``hall_number`` exist for the
    space-group type, the smallest one (the first description of the space-group
    type in International Tables for Crystallography) is chosen. The definition of
    ``hall_number`` is found at :ref:`dataset_spg_get_dataset_spacegroup_type` and
    the corresponding space-group-type information is obtained through
    {func}`get_spacegroup_type`.

    This is expected to work well for the set of symmetry operations whose
    distortion is small. The aim of making this feature is to find
    space-group-type for the set of symmetry operations given by the other
    source than spglib.

    Note that the definition of ``symprec`` is
    different from usual one, but is given in the fractional
    coordinates and so it should be small like ``1e-5``.
    """
    warnings.warn('get_hall_number_from_symmetry() is deprecated. Use get_spacegroup_type_from_symmetry() instead.', DeprecationWarning)
    r = np.array(rotations, dtype='intc', order='C')
    t = np.array(translations, dtype='double', order='C')
    hall_number = _spglib.hall_number_from_symmetry(r, t, symprec)
    return hall_number