import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_share_instance(attrs=None, methods=None):
    """Create a fake share instance

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_instance = {'status': None, 'progress': None, 'share_id': 'share-id-' + uuid.uuid4().hex, 'availability_zone': None, 'replica_state': None, 'created_at': datetime.datetime.now().isoformat(), 'cast_rules_to_readonly': False, 'share_network_id': 'sn-id-' + uuid.uuid4().hex, 'share_server_id': 'ss-id-' + uuid.uuid4().hex, 'host': None, 'access_rules_status': None, 'id': 'instance-id-' + uuid.uuid4().hex}
    share_instance.update(attrs)
    share_instance = osc_fakes.FakeResource(info=copy.deepcopy(share_instance), methods=methods, loaded=True)
    return share_instance