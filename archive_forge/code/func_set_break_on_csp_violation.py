from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
def set_break_on_csp_violation(violation_types: typing.List[CSPViolationType]) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets breakpoint on particular CSP violations.

    **EXPERIMENTAL**

    :param violation_types: CSP Violations to stop upon.
    """
    params: T_JSON_DICT = dict()
    params['violationTypes'] = [i.to_json() for i in violation_types]
    cmd_dict: T_JSON_DICT = {'method': 'DOMDebugger.setBreakOnCSPViolation', 'params': params}
    json = (yield cmd_dict)