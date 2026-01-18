import json
import os
from oslo_utils import encodeutils
from oslo_utils import uuidutils
import prettytable
from osprofiler.cmd import cliutils
from osprofiler.drivers import base
from osprofiler import exc
def datetime_json_serialize(obj):
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        return obj