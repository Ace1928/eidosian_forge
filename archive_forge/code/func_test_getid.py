from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
def test_getid(self):

    class TmpObject(base.Resource):
        id = '4'
    self.assertEqual('4', base.getid(TmpObject(None, {})))