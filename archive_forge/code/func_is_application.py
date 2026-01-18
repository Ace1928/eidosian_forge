from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models.ui import UIElement
from ..util.browser import NEW_PARAM, get_browser_controller
from .notebook import DEFAULT_JUPYTER_URL, ProxyUrlFunc, run_notebook_hook
from .saving import save
from .state import curstate
def is_application(obj: Any) -> TypeGuard[Application]:
    return getattr(obj, '_is_a_bokeh_application_class', False)