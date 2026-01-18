from __future__ import annotations
import argparse
import datetime
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple
from rdflib.graph import Graph  # noqa: E402
from rdflib.namespace import DCTERMS, OWL, RDFS, SKOS  # noqa: E402
from rdflib.util import guess_format  # noqa: E402
from rdflib.namespace import DefinedNamespace, Namespace
def get_target_namespace_elements(g: Graph, target_namespace: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    namespaces = {'dcterms': DCTERMS, 'owl': OWL, 'rdfs': RDFS, 'skos': SKOS}
    q = '\n        SELECT ?s (GROUP_CONCAT(DISTINCT STR(?def)) AS ?defs)\n        WHERE {\n            # all things in the RDF data (anything RDF.type...)\n            ?s a ?o .\n\n            # get any definitions, if they have one\n            OPTIONAL {\n                ?s dcterms:description|rdfs:comment|skos:definition ?def\n            }\n\n            # only get results for the target namespace (supplied by user)\n            FILTER STRSTARTS(STR(?s), "xxx")\n        }\n        GROUP BY ?s\n        '.replace('xxx', target_namespace)
    elements: List[Tuple[str, str]] = []
    for r in g.query(q, initNs=namespaces):
        if TYPE_CHECKING:
            assert isinstance(r, ResultRow)
        elements.append((str(r[0]), str(r[1])))
    elements.sort(key=lambda tup: tup[0])
    elements_strs: List[str] = []
    for e in elements:
        desc = e[1].replace('\n', ' ')
        elements_strs.append(f'    {e[0].replace(target_namespace, '')}: URIRef  # {desc}\n')
    return (elements, elements_strs)