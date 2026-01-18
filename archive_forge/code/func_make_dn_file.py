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
def make_dn_file(output_file_name: Path, target_namespace: str, elements_strs: Iterable[str], object_id: str, fail: bool) -> None:
    header = f'from rdflib.term import URIRef\nfrom rdflib.namespace import DefinedNamespace, Namespace\n\n\nclass {object_id}(DefinedNamespace):\n    """\n    DESCRIPTION_EDIT_ME_!\n\n    Generated from: SOURCE_RDF_FILE_EDIT_ME_!\n    Date: {datetime.datetime.utcnow()}\n    """\n'
    with open(output_file_name, 'w') as f:
        f.write(header)
        f.write('\n')
        f.write(f'    _NS = Namespace("{target_namespace}")')
        f.write('\n\n')
        if fail:
            f.write('    _fail = True')
            f.write('\n\n')
        f.writelines(elements_strs)