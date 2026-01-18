from __future__ import annotations
import re
from typing import Any, Callable
def convert_dict_keys(func: Callable[[str], str], in_dict: dict[Any, Any]) -> dict[Any, Any]:
    """Apply a conversion function to all keys in a dict.

    Parameters
    ----------
    func : callable
        The function to apply. Takes a str and returns a str.
    in_dict : dict
        The dictionary to convert. If some value in this dict is itself a dict,
        it also gets recursively converted.

    Returns
    -------
    dict
        A new dict with all the contents of `in_dict`, but with the keys
        converted by `func`.

    """
    out_dict = dict()
    for k, v in in_dict.items():
        converted_key = func(k)
        if type(v) is dict:
            out_dict[converted_key] = convert_dict_keys(func, v)
        else:
            out_dict[converted_key] = v
    return out_dict