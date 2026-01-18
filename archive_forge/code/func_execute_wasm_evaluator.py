from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def execute_wasm_evaluator(call_frame_id: CallFrameId, evaluator: str, timeout: typing.Optional[runtime.TimeDelta]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[runtime.RemoteObject, typing.Optional[runtime.ExceptionDetails]]]:
    """
    Execute a Wasm Evaluator module on a given call frame.

    **EXPERIMENTAL**

    :param call_frame_id: WebAssembly call frame identifier to evaluate on.
    :param evaluator: Code of the evaluator module.
    :param timeout: **(EXPERIMENTAL)** *(Optional)* Terminate execution after timing out (number of milliseconds).
    :returns: A tuple with the following items:

        0. **result** - Object wrapper for the evaluation result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    """
    params: T_JSON_DICT = dict()
    params['callFrameId'] = call_frame_id.to_json()
    params['evaluator'] = evaluator
    if timeout is not None:
        params['timeout'] = timeout.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.executeWasmEvaluator', 'params': params}
    json = (yield cmd_dict)
    return (runtime.RemoteObject.from_json(json['result']), runtime.ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)