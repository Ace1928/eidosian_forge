from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def real_or_anon_id(r):
    return r._identifier if r._identifier else id_generator.get_anon_id(r)