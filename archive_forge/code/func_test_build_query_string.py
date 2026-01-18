import copy
from mock import Mock
from tests.unit import unittest
from boto.auth import STSAnonHandler
from boto.connection import HTTPRequest
def test_build_query_string(self):
    auth = STSAnonHandler('sts.amazonaws.com', Mock(), self.provider)
    query_string = auth._build_query_string(self.request.params)
    self.assertEqual(query_string, 'Action=AssumeRoleWithWebIdentity' + '&ProviderId=2012-06-01&RoleSessionName=web-identity-federation' + '&Version=2011-06-15&WebIdentityToken=Atza%7CIQEBLjAsAhRkcxQ')