from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
def set_scope_text(style_sheet_id: StyleSheetId, range_: SourceRange, text: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, CSSScope]:
    """
    Modifies the expression of a scope at-rule.

    **EXPERIMENTAL**

    :param style_sheet_id:
    :param range_:
    :param text:
    :returns: The resulting CSS Scope rule after modification.
    """
    params: T_JSON_DICT = dict()
    params['styleSheetId'] = style_sheet_id.to_json()
    params['range'] = range_.to_json()
    params['text'] = text
    cmd_dict: T_JSON_DICT = {'method': 'CSS.setScopeText', 'params': params}
    json = (yield cmd_dict)
    return CSSScope.from_json(json['scope'])