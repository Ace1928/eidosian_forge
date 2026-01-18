from __future__ import print_function, unicode_literals
import argparse
import collections
import io
import json
import logging
import os
import pprint
import re
import sys
import cmakelang
from cmakelang.format import __main__
from cmakelang import configuration
from cmakelang import lex
from cmakelang import parse
from cmakelang.parse.body_nodes import BodyNode
from cmakelang.parse.common import TreeNode
from cmakelang.parse.statement_node import StatementNode
from cmakelang.parse.funs.set import SetFnNode
def process_defn_body(body, out):
    assert isinstance(body, BodyNode)
    variables = {}
    for child in find_statements_in_subtree(body, ['list', 'cmake_parse_arguments', 'set']):
        if isinstance(child, StatementNode):
            if child.get_funname() == 'set':
                process_set_statement(child.argtree, variables)
                continue
            if child.get_funname() == 'list':
                continue
            if child.get_funname() == 'cmake_parse_arguments':
                pargs = child.argtree.get_tokens(kind='semantic')
                pargs.pop(0)
                kwvargs = replace_varrefs(pargs.pop(0).spelling.strip('"'), variables).split(';')
                onevargs = replace_varrefs(pargs.pop(0).spelling.strip('"'), variables).split(';')
                multivargs = replace_varrefs(pargs.pop(0).spelling.strip('"'), variables).split(';')
                pargs = out['pargs']
                nargs = pargs['nargs']
                if nargs == 0:
                    pargs['nargs'] = '*'
                else:
                    pargs['nargs'] = '{}+'.format(nargs)
                pargs['flags'] = flags = []
                for flag in kwvargs:
                    flag = flag.strip().upper()
                    if flag:
                        flags.append(flag)
                out['kwargs'] = kwargs = {}
                for kwarg in onevargs:
                    kwarg = kwarg.strip().upper()
                    if kwarg:
                        kwargs[kwarg] = 1
                for kwarg in multivargs:
                    kwarg = kwarg.strip().upper()
                    if kwarg:
                        kwargs[kwarg] = '+'
                return
    return