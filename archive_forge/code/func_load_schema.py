import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def load_schema(name):
    """
    Load a schema from ./schemas/``name``.json and return it.

    """
    data = pkgutil.get_data('jsonschema', 'schemas/{0}.json'.format(name))
    return json.loads(data.decode('utf-8'))