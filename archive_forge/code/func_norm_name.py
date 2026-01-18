import io
import math
import os
import typing
import weakref
def norm_name(name):
    """Recreate font name that contains PDF hex codes.

            E.g. #20 -> space, chr(32)
            """
    while '#' in name:
        p = name.find('#')
        c = int(name[p + 1:p + 3], 16)
        name = name.replace(name[p:p + 3], chr(c))
    return name