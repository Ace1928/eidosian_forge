from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
def label_context(label_info: strip_label_details, multi_line: bool=True, sep: str=': ') -> strip_label_details:
    """
    Create an unabiguous label string

    If facetting over a single variable, `label_value` is
    used, if two or more variables then `label_both` is used.

    Parameters
    ----------
    label_info : strip_label_details
        Label information
    multi_line : bool
        Whether to place each variable on a separate line
    sep : str
        Separation between variable name and value

    Returns
    -------
    out : str
        Contatenated label values (or pairs of variable names
        & values)
    """
    if len(label_info) == 1:
        return label_value(label_info, multi_line)
    else:
        return label_both(label_info, multi_line, sep)