from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def call_function_on(function_declaration: str, object_id: typing.Optional[RemoteObjectId]=None, arguments: typing.Optional[typing.List[CallArgument]]=None, silent: typing.Optional[bool]=None, return_by_value: typing.Optional[bool]=None, generate_preview: typing.Optional[bool]=None, user_gesture: typing.Optional[bool]=None, await_promise: typing.Optional[bool]=None, execution_context_id: typing.Optional[ExecutionContextId]=None, object_group: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[RemoteObject, typing.Optional[ExceptionDetails]]]:
    """
    Calls function with given declaration on the given object. Object group of the result is
    inherited from the target object.

    :param function_declaration: Declaration of the function to call.
    :param object_id: *(Optional)* Identifier of the object to call function on. Either objectId or executionContextId should be specified.
    :param arguments: *(Optional)* Call arguments. All call arguments must belong to the same JavaScript world as the target object.
    :param silent: *(Optional)* In silent mode exceptions thrown during evaluation are not reported and do not pause execution. Overrides ```setPauseOnException```` state.
    :param return_by_value: *(Optional)* Whether the result is expected to be a JSON object which should be sent by value.
    :param generate_preview: **(EXPERIMENTAL)** *(Optional)* Whether preview should be generated for the result.
    :param user_gesture: *(Optional)* Whether execution should be treated as initiated by user in the UI.
    :param await_promise: *(Optional)* Whether execution should ````await``` for resulting value and return once awaited promise is resolved.
    :param execution_context_id: *(Optional)* Specifies execution context which global object will be used to call function on. Either executionContextId or objectId should be specified.
    :param object_group: *(Optional)* Symbolic group name that can be used to release multiple objects. If objectGroup is not specified and objectId is, objectGroup will be inherited from object.
    :returns: A tuple with the following items:

        0. **result** - Call result.
        1. **exceptionDetails** - *(Optional)* Exception details.
    """
    params: T_JSON_DICT = dict()
    params['functionDeclaration'] = function_declaration
    if object_id is not None:
        params['objectId'] = object_id.to_json()
    if arguments is not None:
        params['arguments'] = [i.to_json() for i in arguments]
    if silent is not None:
        params['silent'] = silent
    if return_by_value is not None:
        params['returnByValue'] = return_by_value
    if generate_preview is not None:
        params['generatePreview'] = generate_preview
    if user_gesture is not None:
        params['userGesture'] = user_gesture
    if await_promise is not None:
        params['awaitPromise'] = await_promise
    if execution_context_id is not None:
        params['executionContextId'] = execution_context_id.to_json()
    if object_group is not None:
        params['objectGroup'] = object_group
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.callFunctionOn', 'params': params}
    json = (yield cmd_dict)
    return (RemoteObject.from_json(json['result']), ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None)