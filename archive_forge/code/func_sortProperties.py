from collections import defaultdict
from functools import cmp_to_key
from rdflib.exceptions import Error
from rdflib.namespace import RDF, RDFS
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
def sortProperties(self, properties):
    """Take a hash from predicate uris to lists of values.
        Sort the lists of values.  Return a sorted list of properties."""
    for prop, objects in properties.items():
        objects.sort(key=cmp_to_key(_object_comparator))
    propList = []
    seen = {}
    for prop in self.predicateOrder:
        if prop in properties and prop not in seen:
            propList.append(prop)
            seen[prop] = True
    props = list(properties.keys())
    props.sort()
    for prop in props:
        if prop not in seen:
            propList.append(prop)
            seen[prop] = True
    return propList