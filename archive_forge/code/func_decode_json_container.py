from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def decode_json_container(jc, bundle):
    if 'prefix' in jc:
        prefixes = jc['prefix']
        for prefix, uri in prefixes.items():
            if prefix != 'default':
                bundle.add_namespace(Namespace(prefix, uri))
            else:
                bundle.set_default_namespace(uri)
        del jc['prefix']
    for rec_type_str in jc:
        rec_type = PROV_RECORD_IDS_MAP[rec_type_str]
        for rec_id, content in jc[rec_type_str].items():
            if hasattr(content, 'items'):
                elements = [content]
            else:
                elements = content
            for element in elements:
                attributes = dict()
                other_attributes = []
                membership_extra_members = None
                for attr_name, values in element.items():
                    attr = PROV_ATTRIBUTES_ID_MAP[attr_name] if attr_name in PROV_ATTRIBUTES_ID_MAP else valid_qualified_name(bundle, attr_name)
                    if attr in PROV_ATTRIBUTES:
                        if isinstance(values, list):
                            if len(values) > 1:
                                if rec_type == PROV_MEMBERSHIP and attr == PROV_ATTR_ENTITY:
                                    membership_extra_members = values[1:]
                                    value = values[0]
                                else:
                                    error_msg = 'The prov package does not support PROV attributes having multiple values.'
                                    logger.error(error_msg)
                                    raise ProvJSONException(error_msg)
                            else:
                                value = values[0]
                        else:
                            value = values
                        value = valid_qualified_name(bundle, value) if attr in PROV_ATTRIBUTE_QNAMES else parse_xsd_datetime(value)
                        attributes[attr] = value
                    elif isinstance(values, list):
                        other_attributes.extend(((attr, decode_json_representation(value, bundle)) for value in values))
                    else:
                        other_attributes.append((attr, decode_json_representation(values, bundle)))
                bundle.new_record(rec_type, rec_id, attributes, other_attributes)
                if membership_extra_members:
                    collection = attributes[PROV_ATTR_COLLECTION]
                    for member in membership_extra_members:
                        bundle.membership(collection, valid_qualified_name(bundle, member))