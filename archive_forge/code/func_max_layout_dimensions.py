from __future__ import unicode_literals
def max_layout_dimensions(dimensions):
    """
    Take the maximum of a list of :class:`.LayoutDimension` instances.
    """
    min_ = max([d.min for d in dimensions if d.min is not None])
    max_ = max([d.max for d in dimensions if d.max is not None])
    preferred = max([d.preferred for d in dimensions])
    return LayoutDimension(min=min_, max=max_, preferred=preferred)