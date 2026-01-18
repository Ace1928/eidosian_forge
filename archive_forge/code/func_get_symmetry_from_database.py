from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_symmetry_from_database(hall_number) -> dict | None:
    """Return symmetry operations corresponding to a Hall symbol. If fails, return None.

    Parameters
    ----------
    hall_number : int
        The Hall symbol is given by the serial number in between 1 and 530.
        The definition of ``hall_number`` is found at
        :ref:`dataset_spg_get_dataset_spacegroup_type`.

    Returns
    -------
    symmetry : dict
        - rotations
            Rotation parts of symmetry operations corresponding to ``hall_number``.
        - translations
            Translation parts of symmetry operations corresponding to ``hall_number``.
    """
    _set_no_error()
    rotations = np.zeros((192, 3, 3), dtype='intc')
    translations = np.zeros((192, 3), dtype='double')
    num_sym = _spglib.symmetry_from_database(rotations, translations, hall_number)
    _set_error_message()
    if num_sym is None:
        return None
    else:
        return {'rotations': np.array(rotations[:num_sym], dtype='intc', order='C'), 'translations': np.array(translations[:num_sym], dtype='double', order='C')}