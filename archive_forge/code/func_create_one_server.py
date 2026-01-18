import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_server(attrs=None, methods=None):
    """Create a fake share server

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_server = {'id': str(uuid.uuid4()), 'project_id': uuid.uuid4().hex, 'updated_at': datetime.datetime.now().isoformat(), 'status': None, 'host': None, 'check_only': False, 'share_network_name': None, 'share_network_id': str(uuid.uuid4()), 'share_network_subnet_id': str(uuid.uuid4()), 'created_at': datetime.datetime.now().isoformat(), 'is_auto_deletable': False, 'identifier': str(uuid.uuid4())}
    share_server.update(attrs)
    share_server = osc_fakes.FakeResource(info=copy.deepcopy(share_server), methods=methods, loaded=True)
    return share_server