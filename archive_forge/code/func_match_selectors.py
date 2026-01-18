from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_selectors(self, el: bs4.Tag, selectors: ct.SelectorList) -> bool:
    """Check if element matches one of the selectors."""
    match = False
    is_not = selectors.is_not
    is_html = selectors.is_html
    if is_html:
        namespaces = self.namespaces
        iframe_restrict = self.iframe_restrict
        self.namespaces = {'html': NS_XHTML}
        self.iframe_restrict = True
    if not is_html or self.is_html:
        for selector in selectors:
            match = is_not
            if isinstance(selector, ct.SelectorNull):
                continue
            if not self.match_tag(el, selector.tag):
                continue
            if selector.flags & ct.SEL_DEFINED and (not self.match_defined(el)):
                continue
            if selector.flags & ct.SEL_ROOT and (not self.match_root(el)):
                continue
            if selector.flags & ct.SEL_SCOPE and (not self.match_scope(el)):
                continue
            if selector.flags & ct.SEL_PLACEHOLDER_SHOWN and (not self.match_placeholder_shown(el)):
                continue
            if not self.match_nth(el, selector.nth):
                continue
            if selector.flags & ct.SEL_EMPTY and (not self.match_empty(el)):
                continue
            if selector.ids and (not self.match_id(el, selector.ids)):
                continue
            if selector.classes and (not self.match_classes(el, selector.classes)):
                continue
            if not self.match_attributes(el, selector.attributes):
                continue
            if selector.flags & RANGES and (not self.match_range(el, selector.flags & RANGES)):
                continue
            if selector.lang and (not self.match_lang(el, selector.lang)):
                continue
            if selector.selectors and (not self.match_subselectors(el, selector.selectors)):
                continue
            if selector.relation and (not self.match_relations(el, selector.relation)):
                continue
            if selector.flags & ct.SEL_DEFAULT and (not self.match_default(el)):
                continue
            if selector.flags & ct.SEL_INDETERMINATE and (not self.match_indeterminate(el)):
                continue
            if selector.flags & DIR_FLAGS and (not self.match_dir(el, selector.flags & DIR_FLAGS)):
                continue
            if selector.contains and (not self.match_contains(el, selector.contains)):
                continue
            match = not is_not
            break
    if is_html:
        self.namespaces = namespaces
        self.iframe_restrict = iframe_restrict
    return match