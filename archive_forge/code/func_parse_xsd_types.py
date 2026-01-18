from collections import defaultdict
from copy import deepcopy
import datetime
import io
import itertools
import logging
import os
import shutil
import tempfile
from urllib.parse import urlparse
import dateutil.parser
from prov import Error, serializers
from prov.constants import *
from prov.identifier import Identifier, QualifiedName, Namespace
def parse_xsd_types(value, datatype):
    return XSD_DATATYPE_PARSERS[datatype](value) if datatype in XSD_DATATYPE_PARSERS else None