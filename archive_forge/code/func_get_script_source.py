from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def get_script_source(script_id: runtime.ScriptId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, typing.Optional[str]]]:
    """
    Returns source for the script with given id.

    :param script_id: Id of the script to get source for.
    :returns: A tuple with the following items:

        0. **scriptSource** - Script source (empty in case of Wasm bytecode).
        1. **bytecode** - *(Optional)* Wasm bytecode.
    """
    params: T_JSON_DICT = dict()
    params['scriptId'] = script_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.getScriptSource', 'params': params}
    json = (yield cmd_dict)
    return (str(json['scriptSource']), str(json['bytecode']) if 'bytecode' in json else None)