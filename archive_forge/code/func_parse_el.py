import copy
import json
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup, FeatureNotFound
from bs4.element import Tag
from . import (
from .dom_helpers import get_attr, get_children, get_descendents, try_urljoin
from .mf_helpers import unordered_list
from .version import __version__
def parse_el(el, ctx):
    """Parse an element for microformats"""
    classes = el.get('class', [])
    potential_microformats = mf2_classes.root(classes, self.filtered_roots)
    if potential_microformats:
        result = handle_microformat(potential_microformats, el)
        ctx.append(result)
    else:
        potential_microformats = backcompat.root(classes)
        if potential_microformats:
            result = handle_microformat(potential_microformats, el, backcompat_mode=True)
            ctx.append(result)
        else:
            for child in get_children(el):
                parse_el(child, ctx)