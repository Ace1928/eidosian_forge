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
def authors(self, rev, who, short=False, sep=None):
    """Generate list of authors, taking --authors option into account.

        The caller has to specify the name of a author list handler,
        as provided by the author list registry, using the ``who``
        argument.  That name only sets a default, though: when the
        user selected a different author list generation using the
        ``--authors`` command line switch, as represented by the
        ``author_list_handler`` constructor argument, that value takes
        precedence.

        :param rev: The revision for which to generate the list of authors.
        :param who: Name of the default handler.
        :param short: Whether to shorten names to either name or address.
        :param sep: What separator to use for automatic concatenation.
        """
    if self._author_list_handler is not None:
        author_list_handler = self._author_list_handler
    else:
        author_list_handler = author_list_registry.get(who)
    names = author_list_handler(rev)
    if short:
        for i in range(len(names)):
            name, address = config.parse_username(names[i])
            if name:
                names[i] = name
            else:
                names[i] = address
    if sep is not None:
        names = sep.join(names)
    return names