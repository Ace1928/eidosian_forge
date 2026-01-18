from __future__ import annotations
import logging # isort:skip
import contextlib
import weakref
from typing import (
from ..core.types import ID
from ..model import Model
from ..util.datatypes import MultiValuedDict
def recompute(self) -> None:
    """ Recompute the set of all models based on references reachable from
        the Document's current roots.

        This computation can be expensive. Use ``freeze`` to wrap operations
        that update the model object graph to avoid over-recompuation

        .. note::
            Any models that remove during recomputation will be noted as
            "previously seen"

        """
    document = self._document()
    if document is None:
        return
    new_models: set[Model] = set()
    for mr in document.roots:
        new_models |= mr.references()
    old_models = set(self._models.values())
    to_detach = old_models - new_models
    to_attach = new_models - old_models
    recomputed: dict[ID, Model] = {}
    recomputed_by_name: MultiValuedDict[str, Model] = MultiValuedDict()
    for mn in new_models:
        recomputed[mn.id] = mn
        if mn.name is not None:
            recomputed_by_name.add_value(mn.name, mn)
    for md in to_detach:
        self._seen_model_ids.add(md.id)
        md._detach_document()
    for ma in to_attach:
        ma._attach_document(document)
        self._new_models.add(ma)
    self._models = recomputed
    self._models_by_name = recomputed_by_name