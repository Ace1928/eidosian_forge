from __future__ import annotations
import json
from typing import Iterable
def as_font(dct):
    if '__gradio_font__' in dct:
        name = dct['name']
        return GoogleFont(name) if dct['class'] == 'google' else Font(name)
    return dct