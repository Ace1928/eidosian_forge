import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
def retrieve_service_content(self):
    ref = vim_util.get_moref(service.SERVICE_INSTANCE, SERVICE_TYPE)
    return self.PbmRetrieveServiceContent(ref)