import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.util import split_into
from sphinx.search.en import SearchEnglish
def get_js_stemmer_rawcodes(self) -> List[str]:
    """Returns a list of non-minified stemmer JS files to copy."""
    if self.lang.js_stemmer_rawcode:
        return [path.join(package_dir, 'search', 'non-minified-js', fname) for fname in ('base-stemmer.js', self.lang.js_stemmer_rawcode)]
    else:
        return []