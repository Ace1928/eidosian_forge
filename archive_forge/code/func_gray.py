import os
from typing import List, Union
@classmethod
def gray(cls, s: str) -> str:
    return cls._format(s, cls._gray)