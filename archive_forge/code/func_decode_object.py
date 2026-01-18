from binascii import hexlify
from collections.abc import MutableMapping
from collections import OrderedDict
from enum import Enum
import itertools
from json import JSONEncoder
from warnings import warn
from fiona.errors import FionaDeprecationWarning
def decode_object(obj):
    """A json.loads object_hook

    Parameters
    ----------
    obj : dict
        A decoded dict.

    Returns
    -------
    Feature, Geometry, or dict

    """
    if isinstance(obj, Object):
        return obj
    else:
        obj = obj.get('__geo_interface__', obj)
        _type = obj.get('type', None)
        if _type == 'Feature' or 'geometry' in obj:
            return Feature.from_dict(obj)
        elif _type in _GEO_TYPES.values():
            return Geometry.from_dict(obj)
        else:
            return obj