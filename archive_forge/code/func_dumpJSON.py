from __future__ import absolute_import, unicode_literals
from builtins import str
import json
from commonmark.node import is_container
def dumpJSON(obj):
    """Output AST in JSON form, this is destructive of block."""
    prepared = prepare(obj)
    return json.dumps(prepared, indent=4, sort_keys=True)