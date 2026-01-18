import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_one_share_group_type(attrs=None, methods=None):
    """Create a fake share group type

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    methods = methods or {}
    share_group_type_info = {'is_public': True, 'group_specs': {'snapshot_support': True}, 'share_types': ['share-types-id-' + uuid.uuid4().hex], 'id': 'share-group-type-id-' + uuid.uuid4().hex, 'name': 'share-group-type-name-' + uuid.uuid4().hex, 'is_default': False}
    share_group_type_info.update(attrs)
    share_group_type = osc_fakes.FakeResource(info=copy.deepcopy(share_group_type_info), methods=methods, loaded=True)
    return share_group_type