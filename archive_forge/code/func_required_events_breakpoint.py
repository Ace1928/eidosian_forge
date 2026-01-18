import types
from _pydev_bundle import pydev_log
from typing import Tuple, Literal
def required_events_breakpoint(self) -> Tuple[Literal['line', 'call'], ...]:
    ret = ()
    for plugin in self.active_plugins:
        new = plugin.required_events_breakpoint()
        if new:
            ret += new
    return ret