from itertools import chain
from matplotlib import cbook, cm, colors as mcolors, markers, image as mimage
from matplotlib.backends.qt_compat import QtGui
from matplotlib.backends.qt_editor import _formlayout
from matplotlib.dates import DateConverter, num2date
def apply_callback(data):
    """A callback to apply changes."""
    orig_limits = {name: getattr(axes, f'get_{name}lim')() for name in axis_map}
    general = data.pop(0)
    curves = data.pop(0) if has_curve else []
    mappables = data.pop(0) if has_sm else []
    if data:
        raise ValueError('Unexpected field')
    title = general.pop(0)
    axes.set_title(title)
    generate_legend = general.pop()
    for i, (name, axis) in enumerate(axis_map.items()):
        axis_min = general[4 * i]
        axis_max = general[4 * i + 1]
        axis_label = general[4 * i + 2]
        axis_scale = general[4 * i + 3]
        if axis.get_scale() != axis_scale:
            getattr(axes, f'set_{name}scale')(axis_scale)
        axis._set_lim(axis_min, axis_max, auto=False)
        axis.set_label_text(axis_label)
        axis.converter = axis_converter[name]
        axis.set_units(axis_units[name])
    for index, curve in enumerate(curves):
        line = labeled_lines[index][1]
        label, linestyle, drawstyle, linewidth, color, marker, markersize, markerfacecolor, markeredgecolor = curve
        line.set_label(label)
        line.set_linestyle(linestyle)
        line.set_drawstyle(drawstyle)
        line.set_linewidth(linewidth)
        rgba = mcolors.to_rgba(color)
        line.set_alpha(None)
        line.set_color(rgba)
        if marker != 'none':
            line.set_marker(marker)
            line.set_markersize(markersize)
            line.set_markerfacecolor(markerfacecolor)
            line.set_markeredgecolor(markeredgecolor)
    for index, mappable_settings in enumerate(mappables):
        mappable = labeled_mappables[index][1]
        if len(mappable_settings) == 5:
            label, cmap, low, high, interpolation = mappable_settings
            mappable.set_interpolation(interpolation)
        elif len(mappable_settings) == 4:
            label, cmap, low, high = mappable_settings
        mappable.set_label(label)
        mappable.set_cmap(cmap)
        mappable.set_clim(*sorted([low, high]))
    if generate_legend:
        draggable = None
        ncols = 1
        if axes.legend_ is not None:
            old_legend = axes.get_legend()
            draggable = old_legend._draggable is not None
            ncols = old_legend._ncols
        new_legend = axes.legend(ncols=ncols)
        if new_legend:
            new_legend.set_draggable(draggable)
    figure = axes.get_figure()
    figure.canvas.draw()
    for name in axis_map:
        if getattr(axes, f'get_{name}lim')() != orig_limits[name]:
            figure.canvas.toolbar.push_current()
            break