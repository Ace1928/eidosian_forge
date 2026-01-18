import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def page_rotation_reset(page, xref, rot, mediabox):
    """Reset page rotation to original values.

    To be used before we return tables."""
    doc = page.parent
    doc.update_stream(xref, b' ')
    page.set_mediabox(mediabox)
    page.set_rotation(rot)
    pno = page.number
    page = doc[pno]
    return page