from typing import List
from ._interfaces import LogTrace
from ._logger import Logger
def formatWithName(obj: object) -> str:
    if hasattr(obj, 'name'):
        return f'{obj} ({obj.name})'
    else:
        return f'{obj}'