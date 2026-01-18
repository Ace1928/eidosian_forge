from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def add_script_to_evaluate_on_load(script_source: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, ScriptIdentifier]:
    """
    Deprecated, please use addScriptToEvaluateOnNewDocument instead.

    **EXPERIMENTAL**

    :param script_source:
    :returns: Identifier of the added script.
    """
    params: T_JSON_DICT = dict()
    params['scriptSource'] = script_source
    cmd_dict: T_JSON_DICT = {'method': 'Page.addScriptToEvaluateOnLoad', 'params': params}
    json = (yield cmd_dict)
    return ScriptIdentifier.from_json(json['identifier'])