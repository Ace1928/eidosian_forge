from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instance_counts(self, **kw):
    return (200, {}, {'instances': [{'module_id': '4321', 'module_name': 'mod1', 'min_date': '2015-05-02T11:06:16', 'max_date': '2015-05-02T11:06:19', 'module_md5': '9db783b92a9355f70c41806659fcb77d', 'current': True, 'count': 1}]})