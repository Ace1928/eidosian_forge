import calendar
from typing import Any, Optional, Tuple
def getArgument(self, name):
    for a in self.methodSignature:
        if a.name == name:
            return a