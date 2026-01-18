from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
@contextmanager
def pruningUnusedNames(varfont):
    from . import log
    origNameIDs = getVariationNameIDs(varfont)
    yield
    log.info('Pruning name table')
    exclude = origNameIDs - getVariationNameIDs(varfont)
    varfont['name'].names[:] = [record for record in varfont['name'].names if record.nameID not in exclude]
    if 'ltag' in varfont:
        if not any((record for record in varfont['name'].names if record.platformID == 0 and record.langID != 65535)):
            del varfont['ltag']