from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
def label_value(label_info: strip_label_details, multi_line: bool=True) -> strip_label_details:
    """
    Keep value as the label

    Parameters
    ----------
    label_info : strip_label_details
        Label information whose values will be returned
    multi_line : bool
        Whether to place each variable on a separate line

    Returns
    -------
    out : strip_label_details
        Label text strings
    """
    label_info = label_info.copy()
    if not multi_line:
        label_info = label_info.collapse()
    return label_info