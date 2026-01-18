import re
from typing import Dict, Any
def child_state(self, src: str):
    child = self.__class__(self)
    child.process(src)
    return child