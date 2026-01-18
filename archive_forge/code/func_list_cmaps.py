import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def list_cmaps(provider=None, records=False, name=None, category=None, source=None, bg=None, reverse=None):
    """
    Return colormap names matching the specified filters.
    """
    available = _list_cmaps(provider=provider, records=True)
    matches = set()
    for avail in available:
        aname = avail.name
        matched = False
        basename = aname[:-2] if aname.endswith('_r') else aname
        if reverse is None or (reverse == True and aname.endswith('_r')) or (reverse == False and (not aname.endswith('_r'))):
            for r in cmap_info:
                if r.name == basename:
                    matched = True
                    r = r._replace(name=aname)
                    if aname.endswith('_r') and r.category != 'Diverging':
                        if r.bg == 'light':
                            r = r._replace(bg='dark')
                        elif r.bg == 'dark':
                            r = r._replace(bg='light')
                    if (name is None or name in r.name) and (provider is None or provider in r.provider) and (category is None or category in r.category) and (source is None or source in r.source) and (bg is None or bg in r.bg):
                        matches.add(r)
            if not matched and (category is None or category == 'Miscellaneous'):
                r = CMapInfo(aname, provider=avail.provider, category='Miscellaneous', source=None, bg=None)
                matches.add(r)
    if records:
        return sorted(matches, key=lambda r: (r.category.split(' ')[-1], r.bg or '', r.name.lower(), r.provider, r.source or ''))
    else:
        return list(unique_iterator(sorted([rec.name for rec in matches], key=lambda n: n.lower())))