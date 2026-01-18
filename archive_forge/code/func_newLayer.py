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
def newLayer(self, name: str, **kwargs: Any) -> Layer:
    """Creates and returns a named layer.

        Args:
            name: The layer name.
            kwargs: Arguments passed to the constructor of Layer.
        """
    if name in self._layers:
        raise KeyError('layer %r already exists' % name)
    self._layers[name] = layer = Layer(name, **kwargs)
    if layer._default:
        self.defaultLayer = layer
    return layer