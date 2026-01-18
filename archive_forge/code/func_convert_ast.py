import sys
from os.path import splitext
from docutils import parsers, nodes
from sphinx import addnodes
from commonmark import Parser
from warnings import warn
def convert_ast(self, ast):
    for node, entering in ast.walker():
        fn_prefix = 'visit' if entering else 'depart'
        fn_name = '{0}_{1}'.format(fn_prefix, node.t.lower())
        fn_default = 'default_{0}'.format(fn_prefix)
        fn = getattr(self, fn_name, None)
        if fn is None:
            fn = getattr(self, fn_default)
        fn(node)