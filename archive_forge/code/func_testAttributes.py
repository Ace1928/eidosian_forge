import unittest
import six
from apitools.base.py.testing import mock
from samples.iam_sample.iam_v1 import iam_v1_client  # nopep8
from samples.iam_sample.iam_v1 import iam_v1_messages  # nopep8
def testAttributes(self):
    inner_classes = set([])
    for key, value in iam_v1_client.IamV1.__dict__.items():
        if isinstance(value, six.class_types):
            inner_classes.add(key)
    self.assertEquals(set(['IamPoliciesService', 'ProjectsService', 'ProjectsServiceAccountsKeysService', 'ProjectsServiceAccountsService', 'RolesService']), inner_classes)