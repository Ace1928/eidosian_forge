import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def get_shares(shares=None, count=2):
    """Get an iterable MagicMock object with a list of faked shares.

        If a shares list is provided, then initialize the Mock object with the
        list. Otherwise create one.
        :param List shares:
            A list of FakeResource objects faking shares
        :param Integer count:
            The number of shares to be faked
        :return
            An iterable Mock object with side_effect set to a list of faked
            shares
        """
    if shares is None:
        shares = FakeShare.create_shares(count)
    return mock.Mock(side_effect=shares)