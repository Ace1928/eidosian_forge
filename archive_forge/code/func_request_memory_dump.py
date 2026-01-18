from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io
def request_memory_dump(deterministic: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[str, bool]]:
    """
    Request a global memory dump.

    :param deterministic: *(Optional)* Enables more deterministic results by forcing garbage collection
    :returns: A tuple with the following items:

        0. **dumpGuid** - GUID of the resulting global memory dump.
        1. **success** - True iff the global memory dump succeeded.
    """
    params: T_JSON_DICT = dict()
    if deterministic is not None:
        params['deterministic'] = deterministic
    cmd_dict: T_JSON_DICT = {'method': 'Tracing.requestMemoryDump', 'params': params}
    json = (yield cmd_dict)
    return (str(json['dumpGuid']), bool(json['success']))