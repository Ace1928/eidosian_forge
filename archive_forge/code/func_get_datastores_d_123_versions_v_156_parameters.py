from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_datastores_d_123_versions_v_156_parameters(self, **kw):
    return (200, {}, {'configuration-parameters': [{'type': 'string', 'name': 'character_set_results', 'datastore_version_id': 'd-123', 'restart_required': 'false'}, {'name': 'connect_timeout', 'min': 2, 'max': 31536000, 'restart_required': 'false', 'type': 'integer', 'datastore_version_id': 'd-123'}, {'type': 'string', 'name': 'character_set_client', 'datastore_version_id': 'd-123', 'restart_required': 'false'}, {'name': 'max_connections', 'min': 1, 'max': 100000, 'restart_required': 'false', 'type': 'integer', 'datastore_version_id': 'd-123'}]})