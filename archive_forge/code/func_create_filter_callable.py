import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
def create_filter_callable(self, args, target, is_expression):
    """write a filter-applying expression based on the filters
        present in the given filter names, adjusting for the global
        'default' filter aliases as needed."""

    def locate_encode(name):
        if re.match('decode\\..+', name):
            return 'filters.' + name
        else:
            return filters.DEFAULT_ESCAPES.get(name, name)
    if 'n' not in args:
        if is_expression:
            if self.compiler.pagetag:
                args = self.compiler.pagetag.filter_args.args + args
            if self.compiler.default_filters and 'n' not in args:
                args = self.compiler.default_filters + args
    for e in args:
        if e == 'n':
            continue
        m = re.match('(.+?)(\\(.*\\))', e)
        if m:
            ident, fargs = m.group(1, 2)
            f = locate_encode(ident)
            e = f + fargs
        else:
            e = locate_encode(e)
            assert e is not None
        target = '%s(%s)' % (e, target)
    return target