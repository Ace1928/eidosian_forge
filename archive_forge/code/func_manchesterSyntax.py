import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def manchesterSyntax(thing, store, boolean=None, transientList=False):
    """
    Core serialization
    thing is a Class and is processed as a subject
    store is an RDFLib Graph to be queried about thing
    """
    assert thing is not None
    if boolean:
        if transientList:
            livechildren = iter(thing)
            children = [manchesterSyntax(child, store) for child in thing]
        else:
            livechildren = iter(Collection(store, thing))
            children = [manchesterSyntax(child, store) for child in Collection(store, thing)]
        if boolean == OWL.intersectionOf:
            childlist = []
            named = []
            for child in livechildren:
                if isinstance(child, URIRef):
                    named.append(child)
                else:
                    childlist.append(child)
            if named:

                def castToQName(x):
                    prefix, uri, localname = store.compute_qname(x)
                    return ':'.join([prefix, localname])
                if len(named) > 1:
                    prefix = '( ' + ' AND '.join(map(castToQName, named)) + ' )'
                else:
                    prefix = manchesterSyntax(named[0], store)
                if childlist:
                    return str(prefix) + ' THAT ' + ' AND '.join([str(manchesterSyntax(x, store)) for x in childlist])
                else:
                    return prefix
            else:
                return '( ' + ' AND '.join([str(c) for c in children]) + ' )'
        elif boolean == OWL.unionOf:
            return '( ' + ' OR '.join([str(c) for c in children]) + ' )'
        elif boolean == OWL.oneOf:
            return '{ ' + ' '.join([str(c) for c in children]) + ' }'
        else:
            assert boolean == OWL.complementOf
    elif OWL.Restriction in store.objects(subject=thing, predicate=RDF.type):
        prop = list(store.objects(subject=thing, predicate=OWL.onProperty))[0]
        prefix, uri, localname = store.compute_qname(prop)
        propstring = ':'.join([prefix, localname])
        label = first(store.objects(subject=prop, predicate=RDFS.label))
        if label:
            propstring = "'%s'" % label
        for onlyclass in store.objects(subject=thing, predicate=OWL.allValuesFrom):
            return '( %s ONLY %s )' % (propstring, manchesterSyntax(onlyclass, store))
        for val in store.objects(subject=thing, predicate=OWL.hasValue):
            return '( %s VALUE %s )' % (propstring, manchesterSyntax(val, store))
        for someclass in store.objects(subject=thing, predicate=OWL.someValuesFrom):
            return '( %s SOME %s )' % (propstring, manchesterSyntax(someclass, store))
        cardlookup = {OWL.maxCardinality: 'MAX', OWL.minCardinality: 'MIN', OWL.cardinality: 'EQUALS'}
        for _s, p, o in store.triples_choices((thing, list(cardlookup.keys()), None)):
            return '( %s %s %s )' % (propstring, cardlookup[p], o)
    compl = list(store.objects(subject=thing, predicate=OWL.complementOf))
    if compl:
        return '( NOT %s )' % manchesterSyntax(compl[0], store)
    else:
        prolog = '\n'.join(['PREFIX %s: <%s>' % (k, nsBinds[k]) for k in nsBinds])
        qstr = prolog + '\nSELECT ?p ?bool WHERE {?class a owl:Class; ?p ?bool .' + '?bool rdf:first ?foo }'
        initb = {Variable('?class'): thing}
        for boolprop, col in store.query(qstr, processor='sparql', initBindings=initb):
            if not isinstance(thing, URIRef):
                return manchesterSyntax(col, store, boolean=boolprop)
        try:
            prefix, uri, localname = store.compute_qname(thing)
            qname = ':'.join([prefix, localname])
        except Exception:
            if isinstance(thing, BNode):
                return thing.n3()
            return thing.identifier if not isinstance(thing, str) else thing
        label = first(Class(thing, graph=store).label)
        if label:
            return label
        else:
            return qname