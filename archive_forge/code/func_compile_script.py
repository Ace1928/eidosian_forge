from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def compile_script(expression: str, source_url: str, persist_script: bool, execution_context_id: typing.Optional[ExecutionContextId]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[typing.Optional[ScriptId], typing.Optional[ExceptionDetails]]]:
    """
    Compiles expression.

    :param expression: Expression to compile.
    :param source_url: Source url to be set for the script.
    :param persist_script: Specifies whether the compiled script should be persisted.
    :param execution_context_id: *(Optional)* Specifies in which execution context to perform script run. If the parameter is omitted the evaluation will be performed in the context of the inspected page.
    :returns: A tuple with the following items:

        0. **scriptId** - *(Optional)* Id of the script.
        1. **exceptionDetails** - *(Optional)* Exception details.
    """
    params: T_JSON_DICT = dict()
    params['expression'] = expression
    params['sourceURL'] = source_url
    params['persistScript'] = persist_script
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.compileScript', 'params': params}
    json = (yield cmd_dict)
    return (ScriptId.from_json(json['scriptId']) if 'scriptId' in json else None, ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)