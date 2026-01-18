import calendar
from io import open
import fs.base
import fs.osfs
from collections.abc import Mapping
from fontTools.ufoLib.utils import numberTypes
def genericDictValidator(value, prototype):
    """
    Generic. (Added at version 3.)
    """
    if not isinstance(value, Mapping):
        return False
    for key, (typ, required) in prototype.items():
        if not required:
            continue
        if key not in value:
            return False
    for key in value.keys():
        if key not in prototype:
            return False
    for key, v in value.items():
        prototypeType, required = prototype[key]
        if v is None and (not required):
            continue
        if not isinstance(v, prototypeType):
            return False
    return True