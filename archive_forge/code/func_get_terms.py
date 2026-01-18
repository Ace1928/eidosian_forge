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
def get_terms(self, fn2index: Dict) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    rvs: Tuple[Dict[str, List[str]], Dict[str, List[str]]] = ({}, {})
    for rv, mapping in zip(rvs, (self._mapping, self._title_mapping)):
        for k, v in mapping.items():
            if len(v) == 1:
                fn, = v
                if fn in fn2index:
                    rv[k] = fn2index[fn]
            else:
                rv[k] = sorted([fn2index[fn] for fn in v if fn in fn2index])
    return rvs