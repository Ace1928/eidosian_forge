from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def ime_set_composition(text: str, selection_start: int, selection_end: int, replacement_start: typing.Optional[int]=None, replacement_end: typing.Optional[int]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    This method sets the current candidate text for ime.
    Use imeCommitComposition to commit the final text.
    Use imeSetComposition with empty string as text to cancel composition.

    **EXPERIMENTAL**

    :param text: The text to insert
    :param selection_start: selection start
    :param selection_end: selection end
    :param replacement_start: *(Optional)* replacement start
    :param replacement_end: *(Optional)* replacement end
    """
    params: T_JSON_DICT = dict()
    params['text'] = text
    params['selectionStart'] = selection_start
    params['selectionEnd'] = selection_end
    if replacement_start is not None:
        params['replacementStart'] = replacement_start
    if replacement_end is not None:
        params['replacementEnd'] = replacement_end
    cmd_dict: T_JSON_DICT = {'method': 'Input.imeSetComposition', 'params': params}
    json = (yield cmd_dict)