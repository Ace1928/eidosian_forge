import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_snapshot(attrs=None, methods=None):
    """Create a fake share snapshot

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_snapshot = {'created_at': datetime.datetime.now().isoformat(), 'description': 'description-' + uuid.uuid4().hex, 'id': 'snapshot-id-' + uuid.uuid4().hex, 'name': 'name-' + uuid.uuid4().hex, 'project_id': 'project-id-' + uuid.uuid4().hex, 'provider_location': None, 'share_id': 'share-id-' + uuid.uuid4().hex, 'share_proto': 'NFS', 'share_size': 1, 'size': 1, 'status': None, 'user_id': 'user-id-' + uuid.uuid4().hex}
    share_snapshot.update(attrs)
    share_snapshot = osc_fakes.FakeResource(info=copy.deepcopy(share_snapshot), methods=methods, loaded=True)
    return share_snapshot