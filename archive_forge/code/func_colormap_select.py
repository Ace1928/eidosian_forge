from __future__ import annotations
def colormap_select(base_colormap, start=0, end=1.0, reverse=False):
    """
    Given a colormap in the form of a list, such as a Bokeh palette,
    return a version of the colormap reversed if requested, and selecting
    a subset (on a scale 0,1.0) of the elements in the colormap list.

    For instance:

    >>> cmap = ["#000000", "#969696", "#d9d9d9", "#ffffff"]
    >>> colormap_select(cmap,reverse=True)
    ['#ffffff', '#d9d9d9', '#969696', '#000000']
    >>> colormap_select(cmap,0.3,reverse=True)
    ['#d9d9d9', '#969696', '#000000']
    """
    full = list(reversed(base_colormap) if reverse else base_colormap)
    num = len(full)
    return full[int(start * num):int(end * num)]