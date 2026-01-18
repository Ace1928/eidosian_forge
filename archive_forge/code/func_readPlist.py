from fontTools.misc.plistlib import dump, dumps, load, loads
from fontTools.misc.textTools import tobytes
from fontTools.ufoLib.utils import deprecated
@deprecated("Use 'fontTools.misc.plistlib.load' instead")
def readPlist(path_or_file):
    did_open = False
    if isinstance(path_or_file, str):
        path_or_file = open(path_or_file, 'rb')
        did_open = True
    try:
        return load(path_or_file, use_builtin_types=False)
    finally:
        if did_open:
            path_or_file.close()