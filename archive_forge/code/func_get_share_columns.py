import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def get_share_columns(share=None):
    """Get the shares columns from a faked shares object.

        :param shares:
            A FakeResource objects faking shares
        :return
            A tuple which may include the following keys:
            ('id', 'name', 'description', 'status', 'size', 'share_type',
             'metadata', 'snapshot', 'availability_zone')
        """
    if share is not None:
        return tuple((k for k in sorted(share.keys())))
    return tuple([])