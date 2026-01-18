from a triple store that implements the Redland storage interface; similarly,
import os
from io import StringIO
from Bio import MissingPythonDependencyError
from Bio.Phylo import CDAO
from ._cdao_owl import cdao_namespaces, resolve_uri
def add_stmt_to_handle(self, handle, stmt):
    """Add URI prefix to handle."""
    stmt_strings = []
    for n, part in enumerate(stmt):
        if isinstance(part, rdflib.URIRef):
            node_uri = str(part)
            changed = False
            for prefix, uri in self.prefixes.items():
                if node_uri.startswith(uri):
                    node_uri = node_uri.replace(uri, f'{prefix}:', 1)
                    if node_uri == 'rdf:type':
                        node_uri = 'a'
                    changed = True
            if changed or ':' in node_uri:
                stmt_strings.append(node_uri)
            else:
                stmt_strings.append(f'<{node_uri}>')
        elif isinstance(part, rdflib.Literal):
            stmt_strings.append(part.n3())
        else:
            stmt_strings.append(str(part))
    handle.write(f'{' '.join(stmt_strings)} .\n')