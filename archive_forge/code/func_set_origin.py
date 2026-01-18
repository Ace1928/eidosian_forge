from collections.abc import Iterator
from typing import Iterable
def set_origin(self, origin: str):
    if super().__repr__() not in self.origins:
        self.origins[super().__repr__()] = origin