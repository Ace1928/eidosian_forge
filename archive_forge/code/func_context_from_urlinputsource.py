from __future__ import annotations
import pathlib
from typing import IO, TYPE_CHECKING, Any, Optional, TextIO, Tuple, Union
from io import TextIOBase, TextIOWrapper
from posixpath import normpath, sep
from urllib.parse import urljoin, urlsplit, urlunsplit
from rdflib.parser import (
def context_from_urlinputsource(source: URLInputSource) -> Optional[str]:
    """
    Please note that JSON-LD documents served with the application/ld+json media type
    MUST have all context information, including references to external contexts,
    within the body of the document. Contexts linked via a
    http://www.w3.org/ns/json-ld#context HTTP Link Header MUST be
    ignored for such documents.
    """
    if source.content_type != 'application/ld+json':
        try:
            links = source.links
        except AttributeError:
            return
        for link in links:
            if ' rel="http://www.w3.org/ns/json-ld#context"' in link:
                i, j = (link.index('<'), link.index('>'))
                if i > -1 and j > -1:
                    return urljoin(source.url, link[i + 1:j])