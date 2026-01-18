import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
@staticmethod
def create_share_backups(attrs=None, count=2):
    """Create multiple fake backups.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param Integer count:
            The number of share backups to be faked
        :return:
            A list of FakeResource objects
        """
    share_backups = []
    for n in range(0, count):
        share_backups.append(FakeShareBackup.create_one_backup(attrs))
    return share_backups