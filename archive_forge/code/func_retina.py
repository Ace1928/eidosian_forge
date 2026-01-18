from __future__ import annotations
from typing import TYPE_CHECKING
def retina(b: bytes):
    display_png(Image(b, format='png', retina=True))