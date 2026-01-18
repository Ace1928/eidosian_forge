from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_security_group_rules(self, body, **kw):
    assert_has_keys(body['security_group_rule'], required=['cidr', 'cidr'])
    return (202, {}, {'security_group_rule': [{'from_port': 3306, 'protocol': 'tcp', 'created': '2015-05-16T17:55:05', 'to_port': 3306, 'security_group_id': '2', 'cidr': '15.0.0.0/24', 'id': 3}]})