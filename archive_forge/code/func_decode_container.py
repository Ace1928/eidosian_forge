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
def decode_container(self, graph, bundle, relation_mapper=relation_mapper, predicate_mapper=predicate_mapper):
    ids = {}
    PROV_CLS_MAP = {}
    formal_attributes = {}
    unique_sets = {}
    for key, val in PROV_BASE_CLS.items():
        PROV_CLS_MAP[key.uri] = PROV_BASE_CLS[key]
    other_attributes = {}
    for stmt in graph.triples((None, RDF.type, None)):
        id = str(stmt[0])
        obj = str(stmt[2])
        if obj in PROV_CLS_MAP:
            if not isinstance(stmt[0], BNode) and self.valid_identifier(id) is None:
                prefix, iri, _ = graph.namespace_manager.compute_qname(id)
                self.document.add_namespace(prefix, iri)
            try:
                prov_obj = PROV_CLS_MAP[obj]
            except AttributeError:
                prov_obj = None
            add_attr = True
            isderivation = pm.PROV['Revision'].uri in stmt[2] or pm.PROV['Quotation'].uri in stmt[2] or pm.PROV['PrimarySource'].uri in stmt[2]
            if id not in ids and prov_obj and (prov_obj.uri == obj or isderivation or isinstance(stmt[0], BNode)):
                ids[id] = prov_obj
                klass = pm.PROV_REC_CLS[prov_obj]
                formal_attributes[id] = OrderedDict([(key, None) for key in klass.FORMAL_ATTRIBUTES])
                unique_sets[id] = OrderedDict([(key, []) for key in klass.FORMAL_ATTRIBUTES])
                add_attr = False or ((isinstance(stmt[0], BNode) or isderivation) and prov_obj.uri != obj)
            if add_attr:
                if id not in other_attributes:
                    other_attributes[id] = []
                obj_formatted = self.decode_rdf_representation(stmt[2], graph)
                other_attributes[id].append((pm.PROV['type'], obj_formatted))
        else:
            if id not in other_attributes:
                other_attributes[id] = []
            obj = self.decode_rdf_representation(stmt[2], graph)
            other_attributes[id].append((pm.PROV['type'], obj))
    for id, pred, obj in graph:
        id = str(id)
        if id not in other_attributes:
            other_attributes[id] = []
        if pred == RDF.type:
            continue
        if pred in relation_mapper:
            if 'alternateOf' in pred:
                getattr(bundle, relation_mapper[pred])(obj, id)
            elif 'mentionOf' in pred:
                mentionBundle = None
                for stmt in graph.triples((URIRef(id), URIRef(pm.PROV['asInBundle'].uri), None)):
                    mentionBundle = stmt[2]
                getattr(bundle, relation_mapper[pred])(id, str(obj), mentionBundle)
            elif 'actedOnBehalfOf' in pred or 'wasAssociatedWith' in pred:
                qualifier = 'qualified' + relation_mapper[pred].upper()[0] + relation_mapper[pred][1:]
                qualifier_bnode = None
                for stmt in graph.triples((URIRef(id), URIRef(pm.PROV[qualifier].uri), None)):
                    qualifier_bnode = stmt[2]
                if qualifier_bnode is None:
                    getattr(bundle, relation_mapper[pred])(id, str(obj))
                else:
                    fakeys = list(formal_attributes[str(qualifier_bnode)].keys())
                    formal_attributes[str(qualifier_bnode)][fakeys[0]] = id
                    formal_attributes[str(qualifier_bnode)][fakeys[1]] = str(obj)
            else:
                getattr(bundle, relation_mapper[pred])(id, str(obj))
        elif id in ids:
            obj1 = self.decode_rdf_representation(obj, graph)
            if obj is not None and obj1 is None:
                raise ValueError(('Error transforming', obj))
            pred_new = pred
            if pred in predicate_mapper:
                pred_new = predicate_mapper[pred]
            if ids[id] == PROV_COMMUNICATION and 'activity' in str(pred_new):
                pred_new = PROV_ATTR_INFORMANT
            if ids[id] == PROV_DELEGATION and 'agent' in str(pred_new):
                pred_new = PROV_ATTR_RESPONSIBLE
            if ids[id] in [PROV_END, PROV_START] and 'entity' in str(pred_new):
                pred_new = PROV_ATTR_TRIGGER
            if ids[id] in [PROV_END] and 'activity' in str(pred_new):
                pred_new = PROV_ATTR_ENDER
            if ids[id] in [PROV_START] and 'activity' in str(pred_new):
                pred_new = PROV_ATTR_STARTER
            if ids[id] == PROV_DERIVATION and 'entity' in str(pred_new):
                pred_new = PROV_ATTR_USED_ENTITY
            if str(pred_new) in [val.uri for val in formal_attributes[id]]:
                qname_key = self.valid_identifier(pred_new)
                formal_attributes[id][qname_key] = obj1
                unique_sets[id][qname_key].append(obj1)
                if len(unique_sets[id][qname_key]) > 1:
                    formal_attributes[id][qname_key] = None
            elif 'qualified' not in str(pred_new) and 'asInBundle' not in str(pred_new):
                other_attributes[id].append((str(pred_new), obj1))
        local_key = str(obj)
        if local_key in ids:
            if 'qualified' in pred:
                formal_attributes[local_key][list(formal_attributes[local_key].keys())[0]] = id
    for id in ids:
        attrs = None
        if id in other_attributes:
            attrs = other_attributes[id]
        items_to_walk = []
        for qname, values in unique_sets[id].items():
            if values and len(values) > 1:
                items_to_walk.append((qname, values))
        if items_to_walk:
            for subset in list(walk(items_to_walk)):
                for key, value in subset.items():
                    formal_attributes[id][key] = value
                bundle.new_record(ids[id], id, formal_attributes[id], attrs)
        else:
            bundle.new_record(ids[id], id, formal_attributes[id], attrs)
        ids[id] = None
        if attrs is not None:
            other_attributes[id] = []
    for key, val in other_attributes.items():
        if val:
            ids[key].add_attributes(val)