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
def normalizeUri(self, rdfTerm: str) -> str:
    """
        Takes an RDF Term and 'normalizes' it into a QName (using the
        registered prefix) or (unlike compute_qname) the Notation 3
        form for URIs: <...URI...>
        """
    try:
        namespace, name = split_uri(rdfTerm)
        if namespace not in self.__strie:
            insert_strie(self.__strie, self.__trie, str(namespace))
        namespace = URIRef(str(namespace))
    except Exception:
        if isinstance(rdfTerm, Variable):
            return '?%s' % rdfTerm
        else:
            return '<%s>' % rdfTerm
    prefix = self.store.prefix(namespace)
    if prefix is None and isinstance(rdfTerm, Variable):
        return '?%s' % rdfTerm
    elif prefix is None:
        return '<%s>' % rdfTerm
    else:
        qNameParts = self.compute_qname(rdfTerm)
        return ':'.join([qNameParts[0], qNameParts[-1]])