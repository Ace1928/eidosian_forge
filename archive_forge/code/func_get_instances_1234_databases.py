from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def get_instances_1234_databases(self, **kw):
    return (200, {}, {'databases': [{'name': 'db_1'}, {'name': 'db_2'}, {'name': 'performance_schema'}]})