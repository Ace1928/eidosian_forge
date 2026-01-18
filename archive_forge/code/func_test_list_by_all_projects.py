from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_list_by_all_projects(self):
    page_mock = mock.Mock()
    self.backups._paginated = page_mock
    all_projects = True
    self.backups.list(all_projects=all_projects)
    page_mock.assert_called_with('/backups', 'backups', None, None, {'all_projects': all_projects})