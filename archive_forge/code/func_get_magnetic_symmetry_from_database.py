from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_magnetic_symmetry_from_database(uni_number, hall_number=0) -> dict | None:
    """Return magnetic symmetry operations from UNI number between 1 and 1651.

    If fails, return None.

    Optionally alternative settings can be specified with Hall number.

    Parameters
    ----------
    uni_number : int
        UNI number between 1 and 1651.
    hall_number : int, optional
        The Hall symbol is given by the serial number in between 1 and 530.

    Returns
    -------
    symmetry : dict
        - 'rotations'
        - 'translations'
        - 'time_reversals'
            0 and 1 indicate ordinary and anti-time-reversal operations, respectively.

    Notes
    -----
    .. versionadded:: 2.0
    """
    _set_no_error()
    rotations = np.zeros((384, 3, 3), dtype='intc')
    translations = np.zeros((384, 3), dtype='double')
    time_reversals = np.zeros(384, dtype='intc')
    num_sym = _spglib.magnetic_symmetry_from_database(rotations, translations, time_reversals, uni_number, hall_number)
    _set_error_message()
    if num_sym is None:
        return None
    else:
        return {'rotations': np.array(rotations[:num_sym], dtype='intc', order='C'), 'translations': np.array(translations[:num_sym], dtype='double', order='C'), 'time_reversals': np.array(time_reversals[:num_sym], dtype='intc', order='C')}