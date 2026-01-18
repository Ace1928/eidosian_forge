import base64
from collections import OrderedDict
import datetime
import io
import dateutil.parser
from rdflib.term import URIRef, BNode
from rdflib.term import Literal as RDFLiteral
from rdflib.graph import ConjunctiveGraph
from rdflib.namespace import RDF, RDFS, XSD
from prov import Error
import prov.model as pm
from prov.constants import (
from prov.serializers import Serializer
def encode_rdf_representation(self, value):
    if isinstance(value, URIRef):
        return value
    elif isinstance(value, pm.Literal):
        return literal_rdf_representation(value)
    elif isinstance(value, datetime.datetime):
        return RDFLiteral(value.isoformat(), datatype=XSD['dateTime'])
    elif isinstance(value, pm.QualifiedName):
        return URIRef(value.uri)
    elif isinstance(value, pm.Identifier):
        return RDFLiteral(value.uri, datatype=XSD['anyURI'])
    elif type(value) in LITERAL_XSDTYPE_MAP:
        return RDFLiteral(value, datatype=LITERAL_XSDTYPE_MAP[type(value)])
    else:
        return RDFLiteral(value)