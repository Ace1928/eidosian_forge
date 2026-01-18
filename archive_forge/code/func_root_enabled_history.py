import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
def root_enabled_history(self, instance):
    """Get root access history of one instance."""
    url = '/mgmt/instances/%s/root' % base.getid(instance)
    resp, body = self.api.client.get(url)
    if not body:
        raise Exception('Call to ' + url + ' did not return a body.')
    return RootHistory(self, body['root_history'])