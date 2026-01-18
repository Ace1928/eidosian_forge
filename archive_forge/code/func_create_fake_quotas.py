import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_fake_quotas(attrs=None):
    """Create a fake quota set

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with project_id, resource and so on
        """
    attrs = attrs or {}
    quotas_info = {'gigabytes': 1000, 'id': 'tenant-id-c96a43119a40ec7d01794cb8', 'share_group_snapshots': 50, 'share_groups': 50, 'share_networks': 10, 'shares': 50, 'shapshot_gigabytes': 1000, 'snapshots': 50, 'per_share_gigabytes': -1}
    quotas_info.update(attrs)
    quotas = osc_fakes.FakeResource(info=copy.deepcopy(quotas_info), loaded=True)
    return quotas