from __future__ import annotations
from fontTools.misc.textTools import byteord, tostr
import re
from bisect import bisect_right
from typing import Literal, TypeVar, overload
from . import Blocks, Scripts, ScriptExtensions, OTTags
def ot_tags_from_script(script_code):
    """Return a list of OpenType script tags associated with a given
    Unicode script code.
    Return ['DFLT'] script tag for invalid/unknown script codes.
    """
    if script_code in OTTags.SCRIPT_EXCEPTIONS:
        return [OTTags.SCRIPT_EXCEPTIONS[script_code]]
    if script_code not in Scripts.NAMES:
        return [OTTags.DEFAULT_SCRIPT]
    script_tags = [script_code[0].lower() + script_code[1:]]
    if script_code in OTTags.NEW_SCRIPT_TAGS:
        script_tags.extend(OTTags.NEW_SCRIPT_TAGS[script_code])
        script_tags.reverse()
    return script_tags