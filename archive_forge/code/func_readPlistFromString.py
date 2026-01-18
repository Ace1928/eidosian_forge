from fontTools.misc.plistlib import dump, dumps, load, loads
from fontTools.misc.textTools import tobytes
from fontTools.ufoLib.utils import deprecated
@deprecated("Use 'fontTools.misc.plistlib.loads' instead")
def readPlistFromString(data):
    return loads(tobytes(data, encoding='utf-8'), use_builtin_types=False)