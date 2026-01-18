from __future__ import annotations
import logging # isort:skip
import math
from collections import defaultdict
from typing import (
from .core.enums import Location, LocationType, SizingModeType
from .core.property.singletons import Undefined, UndefinedType
from .models import (
from .util.dataclasses import dataclass
from .util.warnings import warn
def gridplot(children: list[list[UIElement | None]], *, sizing_mode: SizingModeType | None=None, toolbar_location: LocationType | None='above', ncols: int | None=None, width: int | None=None, height: int | None=None, toolbar_options: dict[ToolbarOptions, Any] | None=None, merge_tools: bool=True) -> GridPlot:
    """ Create a grid of plots rendered on separate canvases.

    The ``gridplot`` function builds a single toolbar for all the plots in the
    grid. ``gridplot`` is designed to layout a set of plots. For general
    grid layout, use the :func:`~bokeh.layouts.layout` function.

    Args:
        children (list of lists of |Plot|): An array of plots to display in a
            grid, given as a list of lists of Plot objects. To leave a position
            in the grid empty, pass None for that position in the children list.
            OR list of |Plot| if called with ncols.

        sizing_mode (``"fixed"``, ``"stretch_both"``, ``"scale_width"``, ``"scale_height"``, ``"scale_both"`` ): How
            will the items in the layout resize to fill the available space.
            Default is ``"fixed"``. For more information on the different
            modes see :attr:`~bokeh.models.LayoutDOM.sizing_mode`
            description on :class:`~bokeh.models.LayoutDOM`.

        toolbar_location (``above``, ``below``, ``left``, ``right`` ): Where the
            toolbar will be located, with respect to the grid. Default is
            ``above``. If set to None, no toolbar will be attached to the grid.

        ncols (int, optional): Specify the number of columns you would like in your grid.
            You must only pass an un-nested list of plots (as opposed to a list of lists of plots)
            when using ncols.

        width (int, optional): The width you would like all your plots to be

        height (int, optional): The height you would like all your plots to be.

        toolbar_options (dict, optional) : A dictionary of options that will be
            used to construct the grid's toolbar (an instance of
            :class:`~bokeh.models.Toolbar`). If none is supplied,
            Toolbar's defaults will be used.

        merge_tools (``True``, ``False``): Combine tools from all child plots into
            a single toolbar.

    Returns:
        GridPlot:

    Examples:

        >>> gridplot([[plot_1, plot_2], [plot_3, plot_4]])
        >>> gridplot([plot_1, plot_2, plot_3, plot_4], ncols=2, width=200, height=100)
        >>> gridplot(
                children=[[plot_1, plot_2], [None, plot_3]],
                toolbar_location='right'
                sizing_mode='fixed',
                toolbar_options=dict(logo='gray')
            )

    """
    if toolbar_options is None:
        toolbar_options = {}
    if toolbar_location:
        if not hasattr(Location, toolbar_location):
            raise ValueError(f'Invalid value of toolbar_location: {toolbar_location}')
    children = _parse_children_arg(children=children)
    if ncols:
        if any((isinstance(child, list) for child in children)):
            raise ValueError('Cannot provide a nested list when using ncols')
        children = list(_chunks(children, ncols))
    if not children:
        children = []
    toolbars: list[Toolbar] = []
    items: list[tuple[UIElement, int, int]] = []
    for y, row in enumerate(children):
        for x, item in enumerate(row):
            if item is None:
                continue
            elif isinstance(item, LayoutDOM):
                if merge_tools:
                    for plot in item.select(dict(type=Plot)):
                        toolbars.append(plot.toolbar)
                        plot.toolbar_location = None
                if width is not None:
                    item.width = width
                if height is not None:
                    item.height = height
                if sizing_mode is not None and _has_auto_sizing(item):
                    item.sizing_mode = sizing_mode
                items.append((item, y, x))
            elif isinstance(item, UIElement):
                continue
            else:
                raise ValueError('Only UIElement and LayoutDOM items can be inserted into a grid')

    def merge(cls: type[Tool], group: list[Tool]) -> Tool | ToolProxy | None:
        if issubclass(cls, (SaveTool, CopyTool, ExamineTool, FullscreenTool)):
            return cls()
        else:
            return None
    tools: list[Tool | ToolProxy] = []
    for toolbar in toolbars:
        tools.extend(toolbar.tools)
    if merge_tools:
        tools = group_tools(tools, merge=merge)
    logos = [toolbar.logo for toolbar in toolbars]
    autohides = [toolbar.autohide for toolbar in toolbars]
    active_drags = [toolbar.active_drag for toolbar in toolbars]
    active_inspects = [toolbar.active_inspect for toolbar in toolbars]
    active_scrolls = [toolbar.active_scroll for toolbar in toolbars]
    active_taps = [toolbar.active_tap for toolbar in toolbars]
    active_multis = [toolbar.active_multi for toolbar in toolbars]
    V = TypeVar('V')

    def assert_unique(values: list[V], name: ToolbarOptions) -> V | UndefinedType:
        if name in toolbar_options:
            return toolbar_options[name]
        n = len(set(values))
        if n == 0:
            return Undefined
        elif n > 1:
            warn(f"found multiple competing values for 'toolbar.{name}' property; using the latest value")
        return values[-1]
    logo = assert_unique(logos, 'logo')
    autohide = assert_unique(autohides, 'autohide')
    active_drag = assert_unique(active_drags, 'active_drag')
    active_inspect = assert_unique(active_inspects, 'active_inspect')
    active_scroll = assert_unique(active_scrolls, 'active_scroll')
    active_tap = assert_unique(active_taps, 'active_tap')
    active_multi = assert_unique(active_multis, 'active_multi')
    toolbar = Toolbar(tools=tools, logo=logo, autohide=autohide, active_drag=active_drag, active_inspect=active_inspect, active_scroll=active_scroll, active_tap=active_tap, active_multi=active_multi)
    gp = GridPlot(children=items, toolbar=toolbar, toolbar_location=toolbar_location, sizing_mode=sizing_mode)
    return gp