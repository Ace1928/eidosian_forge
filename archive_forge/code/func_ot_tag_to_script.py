from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def ot_tag_to_script(tag):
    """Return the Unicode script code for the given OpenType script tag, or
    None for "DFLT" tag or if there is no Unicode script associated with it.
    Raises ValueError if the tag is invalid.
    """
    tag = tostr(tag).strip()
    if not tag or ' ' in tag or len(tag) > 4:
        raise ValueError('invalid OpenType tag: %r' % tag)
    if tag in OTTags.SCRIPT_ALIASES:
        tag = OTTags.SCRIPT_ALIASES[tag]
    while len(tag) != 4:
        tag += str(' ')
    if tag == OTTags.DEFAULT_SCRIPT:
        return None
    if tag in OTTags.NEW_SCRIPT_TAGS_REVERSED:
        return OTTags.NEW_SCRIPT_TAGS_REVERSED[tag]
    if tag in OTTags.SCRIPT_EXCEPTIONS_REVERSED:
        return OTTags.SCRIPT_EXCEPTIONS_REVERSED[tag]
    script_code = tag[0].upper() + tag[1]
    for i in range(2, 4):
        script_code += script_code[i - 1] if tag[i] == ' ' else tag[i]
    if script_code not in Scripts.NAMES:
        return None
    return script_code