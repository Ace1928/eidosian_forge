from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_magnetic_spacegroup_type(uni_number) -> dict | None:
    """Translate UNI number to magnetic space group type information.

    If fails, return None.

    Parameters
    ----------
    uni_number : int
        UNI number between 1 to 1651

    Returns
    -------
    magnetic_spacegroup_type: dict
        See :ref:`api_get_magnetic_spacegroup_type` for these descriptions.

        - uni_number
        - litvin_number
        - bns_number
        - og_number
        - number
        - type

    Notes
    -----
    .. versionadded:: 2.0

    """
    _set_no_error()
    keys = ('uni_number', 'litvin_number', 'bns_number', 'og_number', 'number', 'type')
    msg_type_list = _spglib.magnetic_spacegroup_type(uni_number)
    _set_error_message()
    if msg_type_list is not None:
        msg_type = dict(zip(keys, msg_type_list))
        for key in msg_type:
            if key not in ['uni_number', 'litvin_number', 'number', 'type']:
                msg_type[key] = msg_type[key].strip()
        return msg_type
    else:
        return None