import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def process_index_entry(entry: str, targetid: str) -> List[Tuple[str, str, str, str, Optional[str]]]:
    from sphinx.domains.python import pairindextypes
    indexentries: List[Tuple[str, str, str, str, Optional[str]]] = []
    entry = entry.strip()
    oentry = entry
    main = ''
    if entry.startswith('!'):
        main = 'main'
        entry = entry[1:].lstrip()
    for type in pairindextypes:
        if entry.startswith(type + ':'):
            value = entry[len(type) + 1:].strip()
            value = pairindextypes[type] + '; ' + value
            indexentries.append(('pair', value, targetid, main, None))
            break
    else:
        for type in indextypes:
            if entry.startswith(type + ':'):
                value = entry[len(type) + 1:].strip()
                if type == 'double':
                    type = 'pair'
                indexentries.append((type, value, targetid, main, None))
                break
        else:
            for value in oentry.split(','):
                value = value.strip()
                main = ''
                if value.startswith('!'):
                    main = 'main'
                    value = value[1:].lstrip()
                if not value:
                    continue
                indexentries.append(('single', value, targetid, main, None))
    return indexentries