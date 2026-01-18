from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_exception_details(error_object_id: RemoteObjectId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Optional[ExceptionDetails]]:
    """
    This method tries to lookup and populate exception details for a
    JavaScript Error object.
    Note that the stackTrace portion of the resulting exceptionDetails will
    only be populated if the Runtime domain was enabled at the time when the
    Error was thrown.

    **EXPERIMENTAL**

    :param error_object_id: The error object for which to resolve the exception details.
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['errorObjectId'] = error_object_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Runtime.getExceptionDetails', 'params': params}
    json = (yield cmd_dict)
    return ExceptionDetails.from_json(json['exceptionDetails']) if 'exceptionDetails' in json else None