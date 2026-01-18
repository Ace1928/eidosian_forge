import hashlib
import os.path
import sys
from collections import namedtuple
from copy import copy, deepcopy
import pkgutil
from ast import literal_eval
from contextlib import suppress
from typing import List, Tuple, Union, Callable, Dict, Optional, Sequence, Generator
from .utils import bfs, logger, classify_bool, is_id_continue, is_id_start, bfs_all_unique, small_factors, OrderedSet
from .lexer import Token, TerminalDef, PatternStr, PatternRE, Pattern
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import ParsingFrontend
from .common import LexerConf, ParserConf
from .grammar import RuleOptions, Rule, Terminal, NonTerminal, Symbol, TOKEN_DEFAULT_PRIORITY
from .utils import classify, dedup_list
from .exceptions import GrammarError, UnexpectedCharacters, UnexpectedToken, ParseError, UnexpectedInput
from .tree import Tree, SlottedTree as ST
from .visitors import Transformer, Visitor, v_args, Transformer_InPlace, Transformer_NonRecursive
def load_grammar(self, grammar_text: str, grammar_name: str='<?>', mangle: Optional[Callable[[str], str]]=None) -> None:
    tree = _parse_grammar(grammar_text, grammar_name)
    imports: Dict[Tuple[str, ...], Tuple[Optional[str], Dict[str, str]]] = {}
    for stmt in tree.children:
        if stmt.data == 'import':
            dotted_path, base_path, aliases = self._unpack_import(stmt, grammar_name)
            try:
                import_base_path, import_aliases = imports[dotted_path]
                assert base_path == import_base_path, 'Inconsistent base_path for %s.' % '.'.join(dotted_path)
                import_aliases.update(aliases)
            except KeyError:
                imports[dotted_path] = (base_path, aliases)
    for dotted_path, (base_path, aliases) in imports.items():
        self.do_import(dotted_path, base_path, aliases, mangle)
    for stmt in tree.children:
        if stmt.data in ('term', 'rule'):
            self._define(*self._unpack_definition(stmt, mangle))
        elif stmt.data == 'override':
            r, = stmt.children
            self._define(*self._unpack_definition(r, mangle), override=True)
        elif stmt.data == 'extend':
            r, = stmt.children
            self._extend(*self._unpack_definition(r, mangle))
        elif stmt.data == 'ignore':
            if mangle is None:
                self._ignore(*stmt.children)
        elif stmt.data == 'declare':
            for symbol in stmt.children:
                assert isinstance(symbol, Symbol), symbol
                is_term = isinstance(symbol, Terminal)
                if mangle is None:
                    name = symbol.name
                else:
                    name = mangle(symbol.name)
                self._define(name, is_term, None)
        elif stmt.data == 'import':
            pass
        else:
            assert False, stmt
    term_defs = {name: d.tree for name, d in self._definitions.items() if d.is_term}
    resolve_term_references(term_defs)