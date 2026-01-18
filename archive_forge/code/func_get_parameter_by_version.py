import json
from troveclient import base
from troveclient import common
def get_parameter_by_version(self, version, key):
    """Get a list of valid parameters that can be changed."""
    return self._get('/datastores/versions/%s/parameters/%s' % (version, key))