import copy
import os
import pathlib
import platform
from ctypes import *
from typing import List, Optional, Tuple
import math
import pyglet
from pyglet.font import base
from pyglet.image.codecs.wic import IWICBitmap, WICDecoder, GUID_WICPixelFormat32bppPBGRA
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
@staticmethod
def find_legacy_font(collection: IDWriteFontCollection, font_name: str, bold, italic, stretch, full_debug=False) -> Optional[IDWriteFont]:
    coll_count = collection.GetFontFamilyCount()
    assert _debug_print(f'directwrite: Found {coll_count} fonts in collection.')
    locale = get_system_locale()
    for i in range(coll_count):
        family = IDWriteFontFamily()
        collection.GetFontFamily(i, byref(family))
        family_name_str = IDWriteLocalizedStrings()
        family.GetFamilyNames(byref(family_name_str))
        family_names = Win32DirectWriteFont.unpack_localized_string(family_name_str, locale)
        family_name = family_names[0]
        if family_name[0] != font_name[0]:
            family.Release()
            continue
        assert _debug_print(f'directwrite: Inspecting family name: {family_name}')
        ft_ct = family.GetFontCount()
        face_names = []
        matches = []
        for j in range(ft_ct):
            temp_ft = IDWriteFont()
            family.GetFont(j, byref(temp_ft))
            if _debug_font and full_debug:
                fc_str = IDWriteLocalizedStrings()
                temp_ft.GetFaceNames(byref(fc_str))
                strings = Win32DirectWriteFont.unpack_localized_string(fc_str, locale)
                face_names.extend(strings)
                print(f'directwrite: Face names found: {strings}')
            compat_names = IDWriteLocalizedStrings()
            exists = BOOL()
            temp_ft.GetInformationalStrings(DWRITE_INFORMATIONAL_STRING_WIN32_FAMILY_NAMES, byref(compat_names), byref(exists))
            match_found = False
            if exists.value != 0:
                for compat_name in Win32DirectWriteFont.unpack_localized_string(compat_names, locale):
                    if compat_name == font_name:
                        assert _debug_print(f"Found legacy name '{font_name}' as '{family_name}' in font face '{j}' (collection id #{i}).")
                        match_found = True
                        matches.append((temp_ft.GetWeight(), temp_ft.GetStyle(), temp_ft.GetStretch(), temp_ft))
                        break
            if not match_found:
                temp_ft.Release()
        family.Release()
        if matches:
            write_font = Win32DirectWriteFont.match_closest_font(matches, bold, italic, stretch)
            for match in matches:
                if match[3] != write_font:
                    match[3].Release()
            return write_font
    return None