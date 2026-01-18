from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def get_current_certificates(self):
    """Determine the server certificates that exist on the storage system."""
    if self.cache_get_current_certificates is None:
        current_certificates = []
        try:
            rc, current_certificates = self.request(self.url_path_prefix + 'certificates/server%s' % self.url_path_suffix)
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve server certificates. Array [%s].' % self.ssid)
        self.cache_get_current_certificates = {}
        for certificate in current_certificates:
            certificate.update({'issuer': self.sanitize_distinguished_name(certificate['issuerDN'])})
            self.cache_get_current_certificates.update({self.sanitize_distinguished_name(certificate['subjectDN']): certificate})
    return self.cache_get_current_certificates