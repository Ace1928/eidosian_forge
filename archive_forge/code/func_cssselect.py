import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def cssselect(self, expr, translator='html'):
    """
        Run the CSS expression on this element and its children,
        returning a list of the results.

        Equivalent to lxml.cssselect.CSSSelect(expr, translator='html')(self)
        -- note that pre-compiling the expression can provide a substantial
        speedup.
        """
    from lxml.cssselect import CSSSelector
    return CSSSelector(expr, translator=translator)(self)