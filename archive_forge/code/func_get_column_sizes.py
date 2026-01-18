from __future__ import annotations
import typing
import warnings
from itertools import chain, repeat
import urwid
from urwid.canvas import Canvas, CanvasJoin, CompositeCanvas, SolidCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Align, Sizing, WHSettings
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin, _ContainerElementSizingFlag
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, WidgetError, WidgetWarning
def get_column_sizes(self, size: tuple[int, int] | tuple[int] | tuple[()], focus: bool=False) -> tuple[Sequence[int], Sequence[int], Sequence[tuple[int, int] | tuple[int] | tuple[()]]]:
    """Get column widths, heights and render size parameters"""
    if not size:
        return self._get_fixed_column_sizes(focus=focus)
    widths = tuple(self.column_widths(size=size, focus=focus))
    heights: dict[int, int] = {}
    w_h_args: dict[int, tuple[int, int] | tuple[int] | tuple[()]] = {}
    box: list[int] = []
    box_need_height: list[int] = []
    for i, (width, (widget, (size_kind, _size_weight, is_box))) in enumerate(zip(widths, self.contents)):
        if isinstance(widget, Widget):
            w_sizing = widget.sizing()
        else:
            warnings.warn(f'{widget!r} is not Widget.', ColumnsWarning, stacklevel=3)
            w_sizing = frozenset((Sizing.FLOW, Sizing.BOX))
        if len(size) == 2 and Sizing.BOX in w_sizing:
            heights[i] = size[1]
            w_h_args[i] = (width, size[1])
        elif is_box:
            box.append(i)
        elif Sizing.FLOW in w_sizing:
            if width > 0:
                heights[i] = widget.rows((width,), focus and i == self.focus_position)
            else:
                heights[i] = 0
            w_h_args[i] = (width,)
        elif size_kind == WHSettings.PACK:
            if width > 0:
                heights[i] = widget.pack((), focus and i == self.focus_position)[1]
            else:
                heights[i] = 0
            w_h_args[i] = ()
        else:
            box_need_height.append(i)
    if len(size) == 1:
        if heights:
            max_height = max(heights.values())
            if box_need_height:
                warnings.warn(f'Widgets in columns {box_need_height} ({[self.contents[i][0] for i in box_need_height]}) are BOX widgets not marked "box_columns" while FLOW render is requested (size={size!r})', ColumnsWarning, stacklevel=3)
        else:
            max_height = 1
    else:
        max_height = size[1]
    for idx in (*box, *box_need_height):
        heights[idx] = max_height
        w_h_args[idx] = (widths[idx], max_height)
    return (widths, tuple((heights[idx] for idx in range(len(heights)))), tuple((w_h_args[idx] for idx in range(len(w_h_args)))))