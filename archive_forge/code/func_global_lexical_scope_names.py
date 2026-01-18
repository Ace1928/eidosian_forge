from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def global_lexical_scope_names(execution_context_id: typing.Optional[ExecutionContextId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    Returns all let, const and class variables from global scope.

    :param execution_context_id: *(Optional)* Specifies in which execution context to lookup global scope variables.
    :returns: 
    """
    params: T_JSON_DICT = dict()
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.globalLexicalScopeNames', 'params': params}
    json = (yield cmd_dict)
    return [str(i) for i in json['names']]