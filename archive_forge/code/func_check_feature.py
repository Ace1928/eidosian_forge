from __future__ import annotations
import collections
import os
import sys
import warnings
import PIL
from . import Image
def check_feature(feature):
    """
    Checks if a feature is available.

    :param feature: The feature to check for.
    :returns: ``True`` if available, ``False`` if unavailable, ``None`` if unknown.
    :raises ValueError: If the feature is not defined in this version of Pillow.
    """
    if feature not in features:
        msg = f'Unknown feature {feature}'
        raise ValueError(msg)
    module, flag, ver = features[feature]
    try:
        imported_module = __import__(module, fromlist=['PIL'])
        return getattr(imported_module, flag)
    except ModuleNotFoundError:
        return None
    except ImportError as ex:
        warnings.warn(str(ex))
        return None