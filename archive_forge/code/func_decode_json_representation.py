from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def decode_json_representation(literal, bundle):
    if isinstance(literal, dict):
        value = literal['$']
        datatype = literal['type'] if 'type' in literal else None
        datatype = valid_qualified_name(bundle, datatype)
        langtag = literal['lang'] if 'lang' in literal else None
        if datatype == XSD_ANYURI:
            return Identifier(value)
        elif datatype == PROV_QUALIFIEDNAME:
            return valid_qualified_name(bundle, value)
        else:
            return Literal(value, datatype, langtag)
    else:
        return literal