import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def define_dot_parser(self):
    """Define dot grammar

        Based on the grammar http://www.graphviz.org/doc/info/lang.html
        """
    colon = Literal(':')
    lbrace = Suppress('{')
    rbrace = Suppress('}')
    lbrack = Suppress('[')
    rbrack = Suppress(']')
    lparen = Literal('(')
    rparen = Literal(')')
    equals = Suppress('=')
    comma = Literal(',')
    dot = Literal('.')
    slash = Literal('/')
    bslash = Literal('\\')
    star = Literal('*')
    semi = Suppress(';')
    at = Literal('@')
    minus = Literal('-')
    pluss = Suppress('+')
    strict_ = CaselessLiteral('strict')
    graph_ = CaselessLiteral('graph')
    digraph_ = CaselessLiteral('digraph')
    subgraph_ = CaselessLiteral('subgraph')
    node_ = CaselessLiteral('node')
    edge_ = CaselessLiteral('edge')
    punctuation_ = ''.join([c for c in string.punctuation if c not in '_']) + string.whitespace
    identifier = Word(alphanums + '_').setName('identifier')
    double_quoted_string = Regex('\\"(?:\\\\\\"|\\\\\\\\|[^"])*\\"', re.MULTILINE)
    double_quoted_string.setParseAction(removeQuotes)
    quoted_string = Combine(double_quoted_string + Optional(OneOrMore(pluss + double_quoted_string)), adjacent=False)
    alphastring_ = OneOrMore(CharsNotIn(punctuation_))

    def parse_html(s, loc, toks):
        return '<<%s>>' % ''.join(toks[0])
    opener = '<'
    closer = '>'
    try:
        html_text = pyparsing.nestedExpr(opener, closer, CharsNotIn(opener + closer).setParseAction(lambda t: t[0])).setParseAction(parse_html)
    except:
        log.debug('nestedExpr not available.')
        log.warning('Old version of pyparsing detected. Version 1.4.8 or later is recommended. Parsing of html labels may not work properly.')
        html_text = Combine(Literal('<<') + OneOrMore(CharsNotIn(',]')))
    float_number = Combine(Optional(minus) + OneOrMore(Word(nums + '.'))).setName('float_number')
    ID = (alphastring_ | html_text | float_number | quoted_string | identifier).setName('ID')
    righthand_id = (float_number | ID).setName('righthand_id')
    port_angle = (at + ID).setName('port_angle')
    port_location = (OneOrMore(Group(colon + ID)) | Group(colon + lparen + ID + comma + ID + rparen)).setName('port_location')
    port = Combine(Group(port_location + Optional(port_angle)) | Group(port_angle + Optional(port_location))).setName('port')
    node_id = ID + Optional(port)
    a_list = OneOrMore(ID + Optional(equals + righthand_id) + Optional(comma.suppress())).setName('a_list')
    attr_list = OneOrMore(lbrack + Optional(a_list) + rbrack).setName('attr_list').setResultsName('attrlist')
    attr_stmt = ((graph_ | node_ | edge_) + attr_list).setName('attr_stmt')
    edgeop = (Literal('--') | Literal('->')).setName('edgeop')
    stmt_list = Forward()
    graph_stmt = (lbrace + Optional(stmt_list) + rbrace + Optional(semi)).setName('graph_stmt')
    edge_point = Forward()
    edgeRHS = OneOrMore(edgeop + edge_point)
    edge_stmt = edge_point + edgeRHS + Optional(attr_list)
    subgraph = (Optional(subgraph_, '') + Optional(ID, '') + Group(graph_stmt)).setName('subgraph').setResultsName('ssubgraph')
    edge_point <<= subgraph | graph_stmt | node_id
    node_stmt = (node_id + Optional(attr_list) + Optional(semi)).setName('node_stmt')
    assignment = (ID + equals + righthand_id).setName('assignment')
    stmt = (assignment | edge_stmt | attr_stmt | subgraph | graph_stmt | node_stmt).setName('stmt')
    stmt_list <<= OneOrMore(stmt + Optional(semi))
    graphparser = (Optional(strict_, 'notstrict') + (graph_ | digraph_) + Optional(ID, '') + lbrace + Group(Optional(stmt_list)) + rbrace).setResultsName('graph')
    singleLineComment = Group('//' + restOfLine) | Group('#' + restOfLine)
    graphparser.ignore(singleLineComment)
    graphparser.ignore(cStyleComment)
    node_id.setParseAction(self._proc_node_id)
    assignment.setParseAction(self._proc_attr_assignment)
    a_list.setParseAction(self._proc_attr_list)
    edge_stmt.setParseAction(self._proc_edge_stmt)
    node_stmt.setParseAction(self._proc_node_stmt)
    attr_stmt.setParseAction(self._proc_default_attr_stmt)
    attr_list.setParseAction(self._proc_attr_list_combine)
    subgraph.setParseAction(self._proc_subgraph_stmt)
    graphparser.setParseAction(self._main_graph_stmt)
    return graphparser