from abc import ABC, abstractmethod
import getpass
import sys, os, pickle
import tempfile
import types
import re
from typing import (
from .exceptions import ConfigurationError, assert_config, UnexpectedInput
from .utils import Serialize, SerializeMemoizer, FS, isascii, logger
from .load_grammar import load_grammar, FromPackageLoader, Grammar, verify_used_files, PackageResource, sha256_digest
from .tree import Tree
from .common import LexerConf, ParserConf, _ParserArgType, _LexerArgType
from .lexer import Lexer, BasicLexer, TerminalDef, LexerThread, Token
from .parse_tree_builder import ParseTreeBuilder
from .parser_frontends import _validate_frontend_args, _get_lexer_callbacks, _deserialize_parsing_frontend, _construct_parsing_frontend
from .grammar import Rule
@classmethod
def open_from_package(cls: Type[_T], package: str, grammar_path: str, search_paths: 'Sequence[str]'=[''], **options) -> _T:
    """Create an instance of Lark with the grammar loaded from within the package `package`.
        This allows grammar loading from zipapps.

        Imports in the grammar will use the `package` and `search_paths` provided, through `FromPackageLoader`

        Example:

            Lark.open_from_package(__name__, "example.lark", ("grammars",), parser=...)
        """
    package_loader = FromPackageLoader(package, search_paths)
    full_path, text = package_loader(None, grammar_path)
    options.setdefault('source_path', full_path)
    options.setdefault('import_paths', [])
    options['import_paths'].append(package_loader)
    return cls(text, **options)