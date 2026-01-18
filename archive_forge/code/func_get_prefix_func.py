import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def get_prefix_func(start: int, end: int) -> _GetPrefixFunc:

    def get_prefix(triple: Tuple[str, str, str], context: Optional[str]) -> Generator[str, None, None]:
        if context is None:
            yield ''
        else:
            yield context
        i = start
        while i < end:
            yield triple[i % 3]
            i += 1
        yield ''
    return get_prefix