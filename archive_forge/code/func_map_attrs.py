import re
from lxml import etree, html
def map_attrs(bs_attrs):
    if isinstance(bs_attrs, dict):
        attribs = {}
        for k, v in bs_attrs.items():
            if isinstance(v, list):
                v = ' '.join(v)
            attribs[k] = unescape(v)
    else:
        attribs = {k: unescape(v) for k, v in bs_attrs}
    return attribs