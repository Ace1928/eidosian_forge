import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
@property
def datastore(self):
    return self._datastore_name