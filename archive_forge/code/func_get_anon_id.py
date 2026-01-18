from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def get_anon_id(self, obj, local_prefix='id'):
    if obj not in self._cache:
        self._count += 1
        self._cache[obj] = Identifier('_:%s%d' % (local_prefix, self._count))
    return self._cache[obj]