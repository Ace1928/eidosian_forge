from __future__ import absolute_import, print_function
import sys
import copy
from ruamel.yaml.compat import ordereddict, PY2, string_types, MutableSliceableSequence
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.anchor import Anchor
def dump_comments(d, name='', sep='.', out=sys.stdout):
    """
    recursively dump comments, all but the toplevel preceded by the path
    in dotted form x.0.a
    """
    if isinstance(d, dict) and hasattr(d, 'ca'):
        if name:
            sys.stdout.write('{}\n'.format(name))
        out.write('{}\n'.format(d.ca))
        for k in d:
            dump_comments(d[k], name=name + sep + k if name else k, sep=sep, out=out)
    elif isinstance(d, list) and hasattr(d, 'ca'):
        if name:
            sys.stdout.write('{}\n'.format(name))
        out.write('{}\n'.format(d.ca))
        for idx, k in enumerate(d):
            dump_comments(k, name=name + sep + str(idx) if name else str(idx), sep=sep, out=out)