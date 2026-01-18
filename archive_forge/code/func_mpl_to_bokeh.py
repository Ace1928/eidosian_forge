import numpy as np
from bokeh.core.properties import (
from ...core.options import abbreviated_exception
from ...core.util import arraylike_types
from ...util.transform import dim
from ..util import COLOR_ALIASES, RGB_HEX_REGEX, rgb2hex
def mpl_to_bokeh(properties):
    """
    Utility to process style properties converting any
    matplotlib specific options to their nearest bokeh
    equivalent.
    """
    new_properties = {}
    for k, v in properties.items():
        if isinstance(v, dict):
            new_properties[k] = v
        elif k == 's':
            new_properties['size'] = v
        elif k == 'marker':
            new_properties.update(markers.get(v, {'marker': v}))
        elif (k == 'color' or k.endswith('_color')) and (not isinstance(v, (dict, dim))):
            with abbreviated_exception():
                v = COLOR_ALIASES.get(v, v)
            if isinstance(v, tuple):
                with abbreviated_exception():
                    v = rgb2hex(v)
            new_properties[k] = v
        else:
            new_properties[k] = v
    new_properties.pop('cmap', None)
    return new_properties