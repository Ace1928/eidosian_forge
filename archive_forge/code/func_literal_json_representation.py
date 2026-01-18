from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def literal_json_representation(literal):
    value, datatype, langtag = (literal.value, literal.datatype, literal.langtag)
    if langtag:
        return {'$': value, 'lang': langtag}
    else:
        return {'$': value, 'type': str(datatype)}