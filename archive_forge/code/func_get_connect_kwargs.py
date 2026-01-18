import collections.abc
import copy
import errno
import functools
import http.client
import os
import re
import urllib.parse as urlparse
import osprofiler.web
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import netutils
from glance.common import auth
from glance.common import exception
from glance.common import utils
from glance.i18n import _
def get_connect_kwargs(self):
    connect_kwargs = {'timeout': self.timeout}
    if self.use_ssl:
        if self.key_file is None:
            self.key_file = os.environ.get('GLANCE_CLIENT_KEY_FILE')
        if self.cert_file is None:
            self.cert_file = os.environ.get('GLANCE_CLIENT_CERT_FILE')
        if self.ca_file is None:
            self.ca_file = os.environ.get('GLANCE_CLIENT_CA_FILE')
        if self.cert_file is not None and self.key_file is None:
            msg = _('You have selected to use SSL in connecting, and you have supplied a cert, however you have failed to supply either a key_file parameter or set the GLANCE_CLIENT_KEY_FILE environ variable')
            raise exception.ClientConnectionError(msg)
        if self.key_file is not None and self.cert_file is None:
            msg = _('You have selected to use SSL in connecting, and you have supplied a key, however you have failed to supply either a cert_file parameter or set the GLANCE_CLIENT_CERT_FILE environ variable')
            raise exception.ClientConnectionError(msg)
        if self.key_file is not None and (not os.path.exists(self.key_file)):
            msg = _('The key file you specified %s does not exist') % self.key_file
            raise exception.ClientConnectionError(msg)
        connect_kwargs['key_file'] = self.key_file
        if self.cert_file is not None and (not os.path.exists(self.cert_file)):
            msg = _('The cert file you specified %s does not exist') % self.cert_file
            raise exception.ClientConnectionError(msg)
        connect_kwargs['cert_file'] = self.cert_file
        if self.ca_file is not None and (not os.path.exists(self.ca_file)):
            msg = _('The CA file you specified %s does not exist') % self.ca_file
            raise exception.ClientConnectionError(msg)
        if self.ca_file is None:
            for ca in self.DEFAULT_CA_FILE_PATH.split(':'):
                if os.path.exists(ca):
                    self.ca_file = ca
                    break
        connect_kwargs['ca_file'] = self.ca_file
        connect_kwargs['insecure'] = self.insecure
    return connect_kwargs