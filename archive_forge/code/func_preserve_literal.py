from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.compat import text_type
from ruamel.yaml.anchor import Anchor
def preserve_literal(s):
    return LiteralScalarString(s.replace('\r\n', '\n').replace('\r', '\n'))