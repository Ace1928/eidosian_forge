import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_message(attrs=None):
    """Create a fake message

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    message = {'id': 'message-id-' + uuid.uuid4().hex, 'action_id': '001', 'detail_id': '002', 'user_message': 'user message', 'message_level': 'ERROR', 'resource_type': 'SHARE', 'resource_id': 'resource-id-' + uuid.uuid4().hex, 'created_at': datetime.datetime.now().isoformat(), 'expires_at': (datetime.datetime.now() + datetime.timedelta(days=30)).isoformat(), 'request_id': 'req-' + uuid.uuid4().hex}
    message.update(attrs)
    message = osc_fakes.FakeResource(info=copy.deepcopy(message), loaded=True)
    return message