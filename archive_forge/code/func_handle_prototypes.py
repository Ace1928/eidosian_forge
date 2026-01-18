from rdflib import RDF as ns_rdf
from .. import ns_rdfa
def handle_prototypes(graph):
    to_remove = set()
    for x, ref, PR in graph.triples((None, pref, None)):
        if (PR, ns_rdf['type'], Prototype) in graph:
            to_remove.add((PR, ns_rdf['type'], Prototype))
            to_remove.add((x, ref, PR))
            for PR, p, y in graph.triples((PR, None, None)):
                if not (p == ns_rdf['type'] and y == Prototype):
                    graph.add((x, p, y))
                    to_remove.add((PR, p, y))
    for t in to_remove:
        graph.remove(t)