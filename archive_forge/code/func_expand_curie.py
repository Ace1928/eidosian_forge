import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from unicodedata import category
from urllib.parse import urldefrag, urljoin
from rdflib.term import URIRef, Variable, _is_valid_uri
from rdflib.namespace._BRICK import BRICK
from rdflib.namespace._CSVW import CSVW
from rdflib.namespace._DC import DC
from rdflib.namespace._DCAM import DCAM
from rdflib.namespace._DCAT import DCAT
from rdflib.namespace._DCMITYPE import DCMITYPE
from rdflib.namespace._DCTERMS import DCTERMS
from rdflib.namespace._DOAP import DOAP
from rdflib.namespace._FOAF import FOAF
from rdflib.namespace._GEO import GEO
from rdflib.namespace._ODRL2 import ODRL2
from rdflib.namespace._ORG import ORG
from rdflib.namespace._OWL import OWL
from rdflib.namespace._PROF import PROF
from rdflib.namespace._PROV import PROV
from rdflib.namespace._QB import QB
from rdflib.namespace._RDF import RDF
from rdflib.namespace._RDFS import RDFS
from rdflib.namespace._SDO import SDO
from rdflib.namespace._SH import SH
from rdflib.namespace._SKOS import SKOS
from rdflib.namespace._SOSA import SOSA
from rdflib.namespace._SSN import SSN
from rdflib.namespace._TIME import TIME
from rdflib.namespace._VANN import VANN
from rdflib.namespace._VOID import VOID
from rdflib.namespace._WGS import WGS
from rdflib.namespace._XSD import XSD
def expand_curie(self, curie: str) -> URIRef:
    """
        Expand a CURIE of the form <prefix:element>, e.g. "rdf:type"
        into its full expression:

        >>> import rdflib
        >>> g = rdflib.Graph()
        >>> g.namespace_manager.expand_curie("rdf:type")
        rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')

        Raises exception if a namespace is not bound to the prefix.

        """
    if not type(curie) is str:
        raise TypeError(f'Argument must be a string, not {type(curie).__name__}.')
    parts = curie.split(':', 1)
    if len(parts) != 2:
        raise ValueError('Malformed curie argument, format should be e.g. “foaf:name”.')
    ns = self.store.namespace(parts[0])
    if ns is not None:
        return URIRef(f'{str(ns)}{parts[1]}')
    else:
        raise ValueError(f'Prefix "{curie.split(':')[0]}" not bound to any namespace.')