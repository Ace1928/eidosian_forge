from __future__ import annotations
import logging # isort:skip
import gc
import weakref
from json import loads
from typing import TYPE_CHECKING, Any, Iterable
from jinja2 import Template
from ..core.enums import HoldPolicyType
from ..core.has_props import is_DataModel
from ..core.query import find, is_single_string_selector
from ..core.serialization import (
from ..core.templates import FILE
from ..core.types import ID
from ..core.validation import check_integrity, process_validation_issues
from ..events import Event
from ..model import Model
from ..themes import Theme, built_in_themes, default as default_theme
from ..util.serialization import make_id
from ..util.strings import nice_join
from ..util.version import __version__
from .callbacks import (
from .events import (
from .json import DocJson, PatchJson
from .models import DocumentModelManager
from .modules import DocumentModuleManager
def remove_root(self, model: Model, setter: Setter | None=None) -> None:
    """ Remove a model as root model from this Document.

        Changes to this model may still trigger ``on_change`` callbacks
        on this document, if the model is still referred to by other
        root models.

        Args:
            model (Model) :
                The model to add as a root of this document.

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                In the context of a Bokeh server application, incoming updates
                to properties will be annotated with the session that is
                doing the updating. This value is propagated through any
                subsequent change notifications that the update triggers.
                The session can compare the event setter to itself, and
                suppress any updates that originate from itself.

        """
    if model not in self._roots:
        return
    with self.models.freeze():
        self._roots.remove(model)
    self.callbacks.trigger_on_change(RootRemovedEvent(self, model, setter))