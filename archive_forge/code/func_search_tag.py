import re
import sys
import warnings
from bs4.css import CSS
from bs4.formatter import (
def search_tag(self, markup_name=None, markup_attrs={}):
    """Check whether a Tag with the given name and attributes would
        match this SoupStrainer.

        Used prospectively to decide whether to even bother creating a Tag
        object.

        :param markup_name: A tag name as found in some markup.
        :param markup_attrs: A dictionary of attributes as found in some markup.

        :return: True if the prospective tag would match this SoupStrainer;
            False otherwise.
        """
    found = None
    markup = None
    if isinstance(markup_name, Tag):
        markup = markup_name
        markup_attrs = markup
    if isinstance(self.name, str):
        if markup and (not markup.prefix) and (self.name != markup.name):
            return False
    call_function_with_tag_data = isinstance(self.name, Callable) and (not isinstance(markup_name, Tag))
    if not self.name or call_function_with_tag_data or (markup and self._matches(markup, self.name)) or (not markup and self._matches(markup_name, self.name)):
        if call_function_with_tag_data:
            match = self.name(markup_name, markup_attrs)
        else:
            match = True
            markup_attr_map = None
            for attr, match_against in list(self.attrs.items()):
                if not markup_attr_map:
                    if hasattr(markup_attrs, 'get'):
                        markup_attr_map = markup_attrs
                    else:
                        markup_attr_map = {}
                        for k, v in markup_attrs:
                            markup_attr_map[k] = v
                attr_value = markup_attr_map.get(attr)
                if not self._matches(attr_value, match_against):
                    match = False
                    break
        if match:
            if markup:
                found = markup
            else:
                found = markup_name
    if found and self.string and (not self._matches(found.string, self.string)):
        found = None
    return found