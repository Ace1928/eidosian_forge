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
def renameLayer(self, name: str, newName: str, overwrite: bool=False) -> None:
    """Renames a layer.

        Args:
            name: The old name.
            newName: The new name.
            overwrite: If False, raises exception if newName is already taken. If True,
                overwrites (read: deletes) the old Layer object.
        """
    if name == newName:
        return
    if not overwrite and newName in self._layers:
        raise KeyError('target name %r already exists' % newName)
    layer = self[name]
    del self._layers[name]
    self._layers[newName] = layer
    layer._name = newName
    if newName == DEFAULT_LAYER_NAME:
        self.defaultLayer = layer