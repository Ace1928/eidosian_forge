from troveclient import client as trove_client
from troveclient.v1 import backup_strategy
from troveclient.v1 import backups
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import databases
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
from troveclient.v1 import limits
from troveclient.v1 import management
from troveclient.v1 import metadata
from troveclient.v1 import modules
from troveclient.v1 import quota
from troveclient.v1 import root
from troveclient.v1 import security_groups
from troveclient.v1 import users
from troveclient.v1 import volume_types
def get_database_api_version_from_endpoint(self):
    return self.client.get_database_api_version_from_endpoint()