import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def relative_url(base, other):
    """Return a path to other from base.

    If other is unrelated to base, return other. Else return a relative path.
    This assumes no symlinks as part of the url.
    """
    dummy, base_first_slash = _find_scheme_and_separator(base)
    if base_first_slash is None:
        return other
    dummy, other_first_slash = _find_scheme_and_separator(other)
    if other_first_slash is None:
        return other
    base_scheme = base[:base_first_slash]
    other_scheme = other[:other_first_slash]
    if base_scheme != other_scheme:
        return other
    elif sys.platform == 'win32' and base_scheme == 'file://':
        base_drive = base[base_first_slash + 1:base_first_slash + 3]
        other_drive = other[other_first_slash + 1:other_first_slash + 3]
        if base_drive != other_drive:
            return other
    base_path = base[base_first_slash + 1:]
    other_path = other[other_first_slash + 1:]
    if base_path.endswith('/'):
        base_path = base_path[:-1]
    base_sections = base_path.split('/')
    other_sections = other_path.split('/')
    if base_sections == ['']:
        base_sections = []
    if other_sections == ['']:
        other_sections = []
    output_sections = []
    for b, o in zip(base_sections, other_sections):
        if b != o:
            break
        output_sections.append(b)
    match_len = len(output_sections)
    output_sections = ['..' for x in base_sections[match_len:]]
    output_sections.extend(other_sections[match_len:])
    return '/'.join(output_sections) or '.'