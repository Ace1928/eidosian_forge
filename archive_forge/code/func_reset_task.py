import json
from troveclient import base
from troveclient import common
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
def reset_task(self, cluster_id):
    """Reset the current cluster task to NONE."""
    body = {'reset-task': {}}
    self._action(cluster_id, body)