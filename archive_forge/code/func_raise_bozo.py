from __future__ import annotations
import typing as t
from .exceptions import ListparserError
from .xml_handler import XMLHandler
def raise_bozo(self, error: str) -> None:
    self.harvest['bozo'] = True
    self.harvest['bozo_exception'] = ListparserError(error)