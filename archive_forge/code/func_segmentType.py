from __future__ import annotations
from typing import Optional
from attrs import define
from ufoLib2.serde import serde
@property
def segmentType(self) -> str | None:
    """Returns the type of the point.

        |defcon_compat|
        """
    return self.type