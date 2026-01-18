import logging
from os import mkdir
from os.path import abspath, exists
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.request import pathname2url
from rdflib.store import NO_STORE, VALID_STORE, Store
from rdflib.term import Identifier, Node, URIRef
def bb(u: str) -> bytes:
    return u.encode('utf-8')