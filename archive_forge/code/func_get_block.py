import posixpath
from collections import defaultdict
from django.utils.safestring import mark_safe
from .base import Node, Template, TemplateSyntaxError, TextNode, Variable, token_kwargs
from .library import Library
def get_block(self, name):
    try:
        return self.blocks[name][-1]
    except IndexError:
        return None