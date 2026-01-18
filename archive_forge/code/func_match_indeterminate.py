from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_indeterminate(self, el: bs4.Tag) -> bool:
    """Match default."""
    match = False
    name = cast(str, self.get_attribute_by_name(el, 'name'))

    def get_parent_form(el: bs4.Tag) -> bs4.Tag | None:
        """Find this input's form."""
        form = None
        parent = self.get_parent(el, no_iframe=True)
        while form is None:
            if self.get_tag(parent) == 'form' and self.is_html_tag(parent):
                form = parent
                break
            last_parent = parent
            parent = self.get_parent(parent, no_iframe=True)
            if parent is None:
                form = last_parent
                break
        return form
    form = get_parent_form(el)
    found_form = False
    for f, n, i in self.cached_indeterminate_forms:
        if f is form and n == name:
            found_form = True
            if i is True:
                match = True
            break
    if not found_form:
        checked = False
        for child in self.get_descendants(form, no_iframe=True):
            if child is el:
                continue
            tag_name = self.get_tag(child)
            if tag_name == 'input':
                is_radio = False
                check = False
                has_name = False
                for k, v in self.iter_attributes(child):
                    if util.lower(k) == 'type' and util.lower(v) == 'radio':
                        is_radio = True
                    elif util.lower(k) == 'name' and v == name:
                        has_name = True
                    elif util.lower(k) == 'checked':
                        check = True
                    if is_radio and check and has_name and (get_parent_form(child) is form):
                        checked = True
                        break
            if checked:
                break
        if not checked:
            match = True
        self.cached_indeterminate_forms.append((form, name, match))
    return match