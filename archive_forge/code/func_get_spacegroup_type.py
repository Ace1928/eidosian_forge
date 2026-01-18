from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def get_spacegroup_type(hall_number) -> dict | None:
    """Translate Hall number to space group type information. If it fails, return None.

    This function allows to directly access to the space-group-type database
    in spglib (spg_database.c).
    To specify the space group type with a specific choice, ``hall_number`` is used.
    The definition of ``hall_number`` is found at
    :ref:`dataset_spg_get_dataset_spacegroup_type`.

    Parameters
    ----------
    hall_number : int
        Hall symbol ID.

    Returns
    -------
    spacegroup_type: dict or None
        Dictionary keys are as follows:

        - number : int
            International space group number
        - international_short : str
            International short symbol.
            Equivalent to ``dataset['international']`` of :func:`get_symmetry_dataset`.
        - international_full : str
            International full symbol.
        - international : str
            International symbol.
        - schoenflies : str
            Schoenflies symbol.
        - hall_number : int
            Hall symbol ID number.
        - hall_symbol : str
            Hall symbol.
            Equivalent to ``dataset['hall']`` of `get_symmetry_dataset`,
        - choice : str
            Centring, origin, basis vector setting.
        - pointgroup_international :
            International symbol of crystallographic point group.
            Equivalent to ``dataset['pointgroup_symbol']`` of
            :func:`get_symmetry_dataset`.
        - pointgroup_schoenflies :
            Schoenflies symbol of crystallographic point group.
        - arithmetic_crystal_class_number : int
            Arithmetic crystal class number
        - arithmetic_crystal_class_symbol : str
            Arithmetic crystal class symbol.

    Notes
    -----
    .. versionadded:: 1.9.4

    .. versionchanged:: 2.0
        ``hall_number`` member is added.

    """
    _set_no_error()
    keys = ('number', 'international_short', 'international_full', 'international', 'schoenflies', 'hall_number', 'hall_symbol', 'choice', 'pointgroup_international', 'pointgroup_schoenflies', 'arithmetic_crystal_class_number', 'arithmetic_crystal_class_symbol')
    spg_type_list = _spglib.spacegroup_type(hall_number)
    _set_error_message()
    if spg_type_list is not None:
        spg_type = dict(zip(keys, spg_type_list))
        for key in spg_type:
            if key not in ('number', 'hall_number', 'arithmetic_crystal_class_number'):
                spg_type[key] = spg_type[key].strip()
        return spg_type
    else:
        return None