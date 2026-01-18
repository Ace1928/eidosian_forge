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
def get_js_stemmer_code(self) -> str:
    """Returns JS code that will be inserted into language_data.js."""
    if self.lang.js_stemmer_rawcode:
        js_dir = path.join(package_dir, 'search', 'minified-js')
        with open(path.join(js_dir, 'base-stemmer.js'), encoding='utf-8') as js_file:
            base_js = js_file.read()
        with open(path.join(js_dir, self.lang.js_stemmer_rawcode), encoding='utf-8') as js_file:
            language_js = js_file.read()
        return '%s\n%s\nStemmer = %sStemmer;' % (base_js, language_js, self.lang.language_name)
    else:
        return self.lang.js_stemmer_code