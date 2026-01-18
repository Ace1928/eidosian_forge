from __future__ import annotations
import os
import typing as t
from collections import defaultdict
from functools import update_wrapper
from .. import typing as ft
from .scaffold import _endpoint_from_view_func
from .scaffold import _sentinel
from .scaffold import Scaffold
from .scaffold import setupmethod
def make_setup_state(self, app: App, options: dict[str, t.Any], first_registration: bool=False) -> BlueprintSetupState:
    """Creates an instance of :meth:`~flask.blueprints.BlueprintSetupState`
        object that is later passed to the register callback functions.
        Subclasses can override this to return a subclass of the setup state.
        """
    return BlueprintSetupState(self, app, options, first_registration)