from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from .constants import Sizing
from .overlay import Overlay
from .widget import delegate_to_widget_mixin
from .widget_decoration import WidgetDecoration
def create_pop_up(self) -> Widget:
    """
        Subclass must override this method and return a widget
        to be used for the pop-up.  This method is called once each time
        the pop-up is opened.
        """
    raise NotImplementedError('Subclass must override this method')