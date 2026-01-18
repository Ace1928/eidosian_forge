import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_best_name_string(self, nameID, default_string='', preferred_order=None):
    """
        Retrieve a name string given nameID. Searches available font names
        matching nameID and returns the decoded bytes of the best match.
        "Best" is defined as a preferred list of platform/encoding/languageIDs
        which can be overridden by supplying a preferred_order matching the
        scheme of 'sort_order' (see below).

        The routine will attempt to decode the string's bytes to a Python str, when the
        platform/encoding[/langID] are known (Windows, Mac, or Unicode platforms).
 
        If you prefer more control over name string selection and decoding than
        this routine provides:
            - call self._init_name_string_map()
            - use (nameID, platformID, encodingID, languageID) as a key into 
              the self._name_strings dict
       """
    if not self._name_strings:
        self._init_name_string_map()
    sort_order = preferred_order or ((3, 1, 1033), (1, 0, 0), (0, 6, 0), (0, 4, 0), (0, 3, 0), (0, 2, 0), (0, 1, 0))
    keys_present = [k for k in self._name_strings.keys() if k[0] == nameID]
    if keys_present:
        key_order = {k: v for v, k in enumerate(sort_order)}
        keys_present.sort(key=lambda x: key_order.get(x[1:4]))
        best_key = keys_present[0]
        nsbytes = self._name_strings[best_key]
        if best_key[1:3] == (3, 1) or best_key[1] == 0:
            enc = 'utf-16-be'
        elif best_key[1:4] == (1, 0, 0):
            enc = 'mac-roman'
        else:
            enc = 'unicode_escape'
        ns = nsbytes.decode(enc)
    else:
        ns = default_string
    return ns