import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
def split_compound(compound):
    """
    Split compound formating string into segments.

    >>> split_compound('bold_underline_bright_blue_on_red')
    ['bold', 'underline', 'bright_blue', 'on_red']

    :arg str compound: a string that may contain compounds, separated by
        underline (``_``).
    :rtype: list
    :returns: List of formating string segments
    """
    merged_segs = []
    mergeable_prefixes = ['on', 'bright', 'on_bright']
    for segment in compound.split('_'):
        if merged_segs and merged_segs[-1] in mergeable_prefixes:
            merged_segs[-1] += '_' + segment
        else:
            merged_segs.append(segment)
    return merged_segs