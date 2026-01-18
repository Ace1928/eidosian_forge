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
def parse_stop_word(source: str) -> Set[str]:
    """
    Parse snowball style word list like this:

    * http://snowball.tartarus.org/algorithms/finnish/stop.txt
    """
    result: Set[str] = set()
    for line in source.splitlines():
        line = line.split('|')[0]
        result.update(line.split())
    return result