from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def encode_json_document(document):
    container = encode_json_container(document)
    for bundle in document.bundles:
        bundle_json = encode_json_container(bundle)
        container['bundle'][str(bundle.identifier)] = bundle_json
    return container