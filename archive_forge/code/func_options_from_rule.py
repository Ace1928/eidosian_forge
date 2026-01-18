import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
from io import open
import pkgutil
from ast import literal_eval
from numbers import Integral
from .utils import bfs, Py36, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique
from .lexer import Token, TerminalDef, PatternStr, PatternRE
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol
from .utils import classify, suppress, dedup_list, Str
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
def options_from_rule(name, params, *x):
    if len(x) > 1:
        priority, expansions = x
        priority = int(priority)
    else:
        expansions, = x
        priority = None
    params = [t.value for t in params.children] if params is not None else []
    keep_all_tokens = name.startswith('!')
    name = name.lstrip('!')
    expand1 = name.startswith('?')
    name = name.lstrip('?')
    return (name, params, expansions, RuleOptions(keep_all_tokens, expand1, priority=priority, template_source=name if params else None))