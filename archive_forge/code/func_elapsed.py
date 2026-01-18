import time
from typing import Any
@property
def elapsed(self) -> float:
    return self.stop - self.start