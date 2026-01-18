from __future__ import unicode_literals
def sum_layout_dimensions(dimensions):
    """
    Sum a list of :class:`.LayoutDimension` instances.
    """
    min = sum([d.min for d in dimensions if d.min is not None])
    max = sum([d.max for d in dimensions if d.max is not None])
    preferred = sum([d.preferred for d in dimensions])
    return LayoutDimension(min=min, max=max, preferred=preferred)