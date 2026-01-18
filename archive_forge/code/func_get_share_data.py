import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def get_share_data(share=None):
    """Get the shares data from a faked shares object.

        :param shares:
            A FakeResource objects faking shares
        :return
            A tuple which may include the following values:
            ('ce26708d', 'fake name', 'fake description', 'available',
             20, 'fake share type', "Manila='zorilla', Zorilla='manila',
             Zorilla='zorilla'", 1, 'nova')
        """
    data_list = []
    if share is not None:
        for x in sorted(share.keys()):
            if x == 'tags':
                data_list.append(format_columns.ListColumn(share.info.get(x)))
            else:
                data_list.append(share.info.get(x))
    return tuple(data_list)