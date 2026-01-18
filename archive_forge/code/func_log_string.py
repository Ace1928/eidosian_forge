import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def log_string(self, revno, rev, max_chars, tags=None, prefix=''):
    """Format log info into one string. Truncate tail of string

        :param revno:      revision number or None.
                           Revision numbers counts from 1.
        :param rev:        revision object
        :param max_chars:  maximum length of resulting string
        :param tags:       list of tags or None
        :param prefix:     string to prefix each line
        :return:           formatted truncated string
        """
    out = []
    if revno:
        out.append('%s:' % revno)
    if max_chars is not None:
        out.append(self.truncate(self.short_author(rev), (max_chars + 3) // 4))
    else:
        out.append(self.short_author(rev))
    out.append(self.date_string(rev))
    if len(rev.parent_ids) > 1:
        out.append('[merge]')
    if tags:
        tag_str = '{%s}' % ', '.join(sorted(tags))
        out.append(tag_str)
    out.append(rev.get_summary())
    return self.truncate(prefix + ' '.join(out).rstrip('\n'), max_chars)