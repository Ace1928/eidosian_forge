from __future__ import annotations
import codecs
import os
import pathlib
import sys
from io import BufferedIOBase, BytesIO, RawIOBase, StringIO, TextIOBase, TextIOWrapper
from typing import (
from urllib.parse import urljoin
from urllib.request import Request, url2pathname
from xml.sax import xmlreader
import rdflib.util
from rdflib import __version__
from rdflib._networking import _urlopen
from rdflib.namespace import Namespace
from rdflib.term import URIRef
def get_alternates(self, type_: Optional[str]=None) -> List[str]:
    typestr: Optional[str] = f'type="{type_}"' if type_ else None
    relstr = 'rel="alternate"'
    alts = []
    for link in self.links:
        parts = [p.strip() for p in link.split(';')]
        if relstr not in parts:
            continue
        if typestr:
            if typestr in parts:
                alts.append(parts[0].strip('<>'))
        else:
            alts.append(parts[0].strip('<>'))
    return alts