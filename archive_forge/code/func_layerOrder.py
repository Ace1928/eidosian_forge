from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.errors import Error
from ufoLib2.objects.layer import Layer
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
@layerOrder.setter
def layerOrder(self, order: list[str]) -> None:
    if set(order) != set(self._layers):
        raise Error('`order` must contain the same layers that are currently present.')
    self._layers = {name: self._layers[name] for name in order}