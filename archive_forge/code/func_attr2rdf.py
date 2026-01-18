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
def attr2rdf(attr):
    return URIRef(PROV[PROV_ID_ATTRIBUTES_MAP[attr].split('prov:')[1]].uri)