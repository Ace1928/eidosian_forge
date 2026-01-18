import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import accounts
def test__get_account_name(self):
    account_ = 'account with no name'
    self.assertEqual(account_, accounts.Accounts._get_account_name(account_))
    account_ = mock.Mock()
    account_.name = 'account-name'
    self.assertEqual('account-name', accounts.Accounts._get_account_name(account_))