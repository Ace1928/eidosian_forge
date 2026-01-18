import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def get_extension_masks():
    """Determine which extension support is available by getting the masks."""
    masks = 0
    tr_idx = extension_index(wintab.WTX_TOUCHRING)
    if tr_idx is not None:
        assert _debug('Touchring support found')
        masks |= wtinfo_uint(wintab.WTI_EXTENSIONS + tr_idx, wintab.EXT_MASK)
    else:
        assert _debug('Touchring extension not found.')
    ts_idx = extension_index(wintab.WTX_TOUCHSTRIP)
    if ts_idx is not None:
        assert _debug('Touchstrip support found.')
        masks |= wtinfo_uint(wintab.WTI_EXTENSIONS + ts_idx, wintab.EXT_MASK)
    else:
        assert _debug('Touchstrip extension not found.')
    expkeys_idx = extension_index(wintab.WTX_EXPKEYS2)
    if expkeys_idx is not None:
        assert _debug('ExpressKey support found.')
        masks |= wtinfo_uint(wintab.WTI_EXTENSIONS + expkeys_idx, wintab.EXT_MASK)
    else:
        assert _debug('ExpressKey extension not found.')
    return masks