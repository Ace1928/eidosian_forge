from .base import Style, DEFAULT_ATTRS, ANSI_COLOR_NAMES
from .defaults import DEFAULT_STYLE_EXTENSIONS
from .utils import merge_attrs, split_token_in_parts
from six.moves import range
def get_attrs_for_token(self, token):
    list_of_attrs = []
    for token in split_token_in_parts(token):
        list_of_attrs.append(self.token_to_attrs.get(token, DEFAULT_ATTRS))
    return merge_attrs(list_of_attrs)