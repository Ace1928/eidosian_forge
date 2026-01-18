import unittest
import httplib2
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py.testing import mock
from samples.fusiontables_sample.fusiontables_v1 import \
from samples.fusiontables_sample.fusiontables_v1 import \
def testMockPreservesServiceMethods(self):
    services = _GetApiServices(fusiontables.FusiontablesV1)
    with mock.Client(fusiontables.FusiontablesV1):
        mocked_services = _GetApiServices(fusiontables.FusiontablesV1)
        self.assertEquals(services.keys(), mocked_services.keys())
        for name, service in six.iteritems(services):
            mocked_service = mocked_services[name]
            methods = service.GetMethodsList()
            for method in methods:
                mocked_method = getattr(mocked_service, method)
                mocked_method_config = mocked_method.method_config()
                method_config = getattr(service, method).method_config()
                self.assertEquals(method_config, mocked_method_config)