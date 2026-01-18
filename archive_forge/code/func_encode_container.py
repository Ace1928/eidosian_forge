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
def encode_container(self, bundle, PROV_N_MAP=PROV_N_MAP, container=None, identifier=None):
    if container is None:
        container = ConjunctiveGraph(identifier=identifier)
        nm = container.namespace_manager
        nm.bind('prov', PROV.uri)
    for namespace in bundle.namespaces:
        container.bind(namespace.prefix, namespace.uri)
    id_generator = AnonymousIDGenerator()
    real_or_anon_id = lambda record: record._identifier.uri if record._identifier else id_generator.get_anon_id(record)
    for record in bundle._records:
        rec_type = record.get_type()
        if hasattr(record, 'identifier') and record.identifier:
            identifier = URIRef(str(real_or_anon_id(record)))
            container.add((identifier, RDF.type, URIRef(rec_type.uri)))
        else:
            identifier = None
        if record.attributes:
            bnode = None
            formal_objects = []
            used_objects = []
            all_attributes = list(record.formal_attributes) + list(record.attributes)
            formal_qualifiers = False
            for attrid, (attr, value) in enumerate(list(record.formal_attributes)):
                if identifier is not None and value is not None or (identifier is None and value is not None and (attrid > 1)):
                    formal_qualifiers = True
            has_qualifiers = len(record.extra_attributes) > 0 or formal_qualifiers
            for idx, (attr, value) in enumerate(all_attributes):
                if record.is_relation():
                    pred = URIRef(PROV[PROV_N_MAP[rec_type]].uri)
                    if bnode is None:
                        valid_formal_indices = set()
                        for idx, (key, val) in enumerate(record.formal_attributes):
                            formal_objects.append(key)
                            if val:
                                valid_formal_indices.add(idx)
                        used_objects = [record.formal_attributes[0][0]]
                        subj = None
                        if record.formal_attributes[0][1]:
                            subj = URIRef(record.formal_attributes[0][1].uri)
                        if identifier is None and subj is not None:
                            try:
                                obj_val = record.formal_attributes[1][1]
                                obj_attr = URIRef(record.formal_attributes[1][0].uri)
                            except IndexError:
                                obj_val = None
                            if obj_val and (rec_type not in {PROV_END, PROV_START, PROV_USAGE, PROV_GENERATION, PROV_DERIVATION, PROV_ASSOCIATION, PROV_INVALIDATION} or (valid_formal_indices == {0, 1} and len(record.extra_attributes) == 0)):
                                used_objects.append(record.formal_attributes[1][0])
                                obj_val = self.encode_rdf_representation(obj_val)
                                if rec_type == PROV_ALTERNATE:
                                    subj, obj_val = (obj_val, subj)
                                container.add((subj, pred, obj_val))
                                if rec_type == PROV_MENTION:
                                    if record.formal_attributes[2][1]:
                                        used_objects.append(record.formal_attributes[2][0])
                                        obj_val = self.encode_rdf_representation(record.formal_attributes[2][1])
                                        container.add((subj, URIRef(PROV['asInBundle'].uri), obj_val))
                                    has_qualifiers = False
                        if rec_type in [PROV_ALTERNATE]:
                            continue
                        if subj and (has_qualifiers or identifier):
                            qualifier = rec_type._localpart
                            rec_uri = rec_type.uri
                            for attr_name, val in record.extra_attributes:
                                if attr_name == PROV['type']:
                                    if PROV['Revision'] == val or PROV['Quotation'] == val or PROV['PrimarySource'] == val:
                                        qualifier = val._localpart
                                        rec_uri = val.uri
                                        if identifier is not None:
                                            container.remove((identifier, RDF.type, URIRef(rec_type.uri)))
                            QRole = URIRef(PROV['qualified' + qualifier].uri)
                            if identifier is not None:
                                container.add((subj, QRole, identifier))
                            else:
                                bnode = identifier = BNode()
                                container.add((subj, QRole, identifier))
                                container.add((identifier, RDF.type, URIRef(rec_uri)))
                    if value is not None and attr not in used_objects:
                        if attr in formal_objects:
                            pred = attr2rdf(attr)
                        elif attr == PROV['role']:
                            pred = URIRef(PROV['hadRole'].uri)
                        elif attr == PROV['plan']:
                            pred = URIRef(PROV['hadPlan'].uri)
                        elif attr == PROV['type']:
                            pred = RDF.type
                        elif attr == PROV['label']:
                            pred = RDFS.label
                        elif isinstance(attr, pm.QualifiedName):
                            pred = URIRef(attr.uri)
                        else:
                            pred = self.encode_rdf_representation(attr)
                        if PROV['plan'].uri in pred:
                            pred = URIRef(PROV['hadPlan'].uri)
                        if PROV['informant'].uri in pred:
                            pred = URIRef(PROV['activity'].uri)
                        if PROV['responsible'].uri in pred:
                            pred = URIRef(PROV['agent'].uri)
                        if rec_type == PROV_DELEGATION and PROV['activity'].uri in pred:
                            pred = URIRef(PROV['hadActivity'].uri)
                        if rec_type in [PROV_END, PROV_START] and PROV['trigger'].uri in pred or (rec_type in [PROV_USAGE] and PROV['used'].uri in pred):
                            pred = URIRef(PROV['entity'].uri)
                        if rec_type in [PROV_GENERATION, PROV_END, PROV_START, PROV_USAGE, PROV_INVALIDATION]:
                            if PROV['time'].uri in pred:
                                pred = URIRef(PROV['atTime'].uri)
                            if PROV['ender'].uri in pred:
                                pred = URIRef(PROV['hadActivity'].uri)
                            if PROV['starter'].uri in pred:
                                pred = URIRef(PROV['hadActivity'].uri)
                            if PROV['location'].uri in pred:
                                pred = URIRef(PROV['atLocation'].uri)
                        if rec_type in [PROV_ACTIVITY]:
                            if PROV_ATTR_STARTTIME in pred:
                                pred = URIRef(PROV['startedAtTime'].uri)
                            if PROV_ATTR_ENDTIME in pred:
                                pred = URIRef(PROV['endedAtTime'].uri)
                        if rec_type == PROV_DERIVATION:
                            if PROV['activity'].uri in pred:
                                pred = URIRef(PROV['hadActivity'].uri)
                            if PROV['generation'].uri in pred:
                                pred = URIRef(PROV['hadGeneration'].uri)
                            if PROV['usage'].uri in pred:
                                pred = URIRef(PROV['hadUsage'].uri)
                            if PROV['usedEntity'].uri in pred:
                                pred = URIRef(PROV['entity'].uri)
                        container.add((identifier, pred, self.encode_rdf_representation(value)))
                    continue
                if value is None:
                    continue
                if isinstance(value, pm.ProvRecord):
                    obj = URIRef(str(real_or_anon_id(value)))
                else:
                    obj = self.encode_rdf_representation(value)
                if attr == PROV['location']:
                    pred = URIRef(PROV['atLocation'].uri)
                    if False and isinstance(value, (URIRef, pm.QualifiedName)):
                        if isinstance(value, pm.QualifiedName):
                            value = URIRef(value.uri)
                        container.add((identifier, pred, value))
                    else:
                        container.add((identifier, pred, self.encode_rdf_representation(obj)))
                    continue
                if attr == PROV['type']:
                    pred = RDF.type
                elif attr == PROV['label']:
                    pred = RDFS.label
                elif attr == PROV_ATTR_STARTTIME:
                    pred = URIRef(PROV['startedAtTime'].uri)
                elif attr == PROV_ATTR_ENDTIME:
                    pred = URIRef(PROV['endedAtTime'].uri)
                else:
                    pred = self.encode_rdf_representation(attr)
                container.add((identifier, pred, obj))
    return container