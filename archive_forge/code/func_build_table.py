import sys
import re
from types import FunctionType, MethodType
from docutils import nodes, statemachine, utils
from docutils import ApplicationError, DataError
from docutils.statemachine import StateMachineWS, StateWS
from docutils.nodes import fully_normalize_name as normalize_name
from docutils.nodes import whitespace_normalize_name
import docutils.parsers.rst
from docutils.parsers.rst import directives, languages, tableparser, roles
from docutils.parsers.rst.languages import en as _fallback_language_module
from docutils.utils import escape2null, unescape, column_width
from docutils.utils import punctuation_chars, roman, urischemes
from docutils.utils import split_escaped_whitespace
def build_table(self, tabledata, tableline, stub_columns=0, widths=None):
    colwidths, headrows, bodyrows = tabledata
    table = nodes.table()
    if widths == 'auto':
        table['classes'] += ['colwidths-auto']
    elif widths:
        table['classes'] += ['colwidths-given']
    tgroup = nodes.tgroup(cols=len(colwidths))
    table += tgroup
    for colwidth in colwidths:
        colspec = nodes.colspec(colwidth=colwidth)
        if stub_columns:
            colspec.attributes['stub'] = 1
            stub_columns -= 1
        tgroup += colspec
    if headrows:
        thead = nodes.thead()
        tgroup += thead
        for row in headrows:
            thead += self.build_table_row(row, tableline)
    tbody = nodes.tbody()
    tgroup += tbody
    for row in bodyrows:
        tbody += self.build_table_row(row, tableline)
    return table