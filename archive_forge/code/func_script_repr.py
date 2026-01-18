from param.parameterized import get_occupied_slots
from .util import datetime_types
def script_repr(self, imports=None, prefix='    '):
    if imports is None:
        imports = []
    cls = self.__class__.__name__
    mod = self.__module__
    imports.append(f'from {mod} import {cls}')
    return self.__str__()