from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .overlay import Overlay
from .widget import delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
def get_pop_up_parameters(self) -> PopUpParametersModel:
    """
        Subclass must override this method and have it return a dict, eg:

        {'left':0, 'top':1, 'overlay_width':30, 'overlay_height':4}

        This method is called each time this widget is rendered.
        """
    raise NotImplementedError('Subclass must override this method')