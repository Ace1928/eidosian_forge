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
def do_import(self, dotted_path: Tuple[str, ...], base_path: Optional[str], aliases: Dict[str, str], base_mangle: Optional[Callable[[str], str]]=None) -> None:
    assert dotted_path
    mangle = _get_mangle('__'.join(dotted_path), aliases, base_mangle)
    grammar_path = os.path.join(*dotted_path) + EXT
    to_try = self.import_paths + ([base_path] if base_path is not None else []) + [stdlib_loader]
    for source in to_try:
        try:
            if callable(source):
                joined_path, text = source(base_path, grammar_path)
            else:
                joined_path = os.path.join(source, grammar_path)
                with open(joined_path, encoding='utf8') as f:
                    text = f.read()
        except IOError:
            continue
        else:
            h = sha256_digest(text)
            if self.used_files.get(joined_path, h) != h:
                raise RuntimeError('Grammar file was changed during importing')
            self.used_files[joined_path] = h
            gb = GrammarBuilder(self.global_keep_all_tokens, self.import_paths, self.used_files)
            gb.load_grammar(text, joined_path, mangle)
            gb._remove_unused(map(mangle, aliases))
            for name in gb._definitions:
                if name in self._definitions:
                    raise GrammarError("Cannot import '%s' from '%s': Symbol already defined." % (name, grammar_path))
            self._definitions.update(**gb._definitions)
            break
    else:
        open(grammar_path, encoding='utf8')
        assert False, "Couldn't import grammar %s, but a corresponding file was found at a place where lark doesn't search for it" % (dotted_path,)