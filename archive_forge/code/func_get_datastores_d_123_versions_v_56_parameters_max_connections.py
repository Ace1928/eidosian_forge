from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_datastores_d_123_versions_v_56_parameters_max_connections(self, **kw):
    r = self.get_datastores_d_123_versions_v_156_parameters()[2]['configuration-parameters'][3]
    return (200, {}, r)