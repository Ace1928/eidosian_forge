from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
def get_possible_breakpoints(start: Location, end: typing.Optional[Location]=None, restrict_to_function: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[BreakLocation]]:
    """
    Returns possible locations for breakpoint. scriptId in start and end range locations should be
    the same.

    :param start: Start of range to search possible breakpoint locations in.
    :param end: *(Optional)* End of range to search possible breakpoint locations in (excluding). When not specified, end of scripts is used as end of range.
    :param restrict_to_function: *(Optional)* Only consider locations which are in the same (non-nested) function as start.
    :returns: List of the possible breakpoint locations.
    """
    params: T_JSON_DICT = dict()
    params['start'] = start.to_json()
    if end is not None:
        params['end'] = end.to_json()
    if restrict_to_function is not None:
        params['restrictToFunction'] = restrict_to_function
    cmd_dict: T_JSON_DICT = {'method': 'Debugger.getPossibleBreakpoints', 'params': params}
    json = (yield cmd_dict)
    return [BreakLocation.from_json(i) for i in json['locations']]