from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_datastores_mysql_versions_some_version_id_volume_types(self, **kw):
    return self.get_volume_types()