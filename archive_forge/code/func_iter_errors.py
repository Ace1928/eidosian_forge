import hashlib
import os
from typing import Generic, TypeVar, Union, Dict, Optional, Any
from pathlib import Path
from parso._compatibility import is_pypy
from parso.pgen2 import generate_grammar
from parso.utils import split_lines, python_bytes_to_unicode, \
from parso.python.diff import DiffParser
from parso.python.tokenize import tokenize_lines, tokenize
from parso.python.token import PythonTokenTypes
from parso.cache import parser_cache, load_module, try_to_save_module
from parso.parser import BaseParser
from parso.python.parser import Parser as PythonParser
from parso.python.errors import ErrorFinderConfig
from parso.python import pep8
from parso.file_io import FileIO, KnownContentFileIO
from parso.normalizer import RefactoringNormalizer, NormalizerConfig
def iter_errors(self, node):
    """
        Given a :py:class:`parso.tree.NodeOrLeaf` returns a generator of
        :py:class:`parso.normalizer.Issue` objects. For Python this is
        a list of syntax/indentation errors.
        """
    if self._error_normalizer_config is None:
        raise ValueError('No error normalizer specified for this grammar.')
    return self._get_normalizer_issues(node, self._error_normalizer_config)