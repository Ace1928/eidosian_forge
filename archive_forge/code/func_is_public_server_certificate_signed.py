from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def is_public_server_certificate_signed(self):
    """Return whether the public server certificate is signed."""
    if self.cache_is_public_server_certificate_signed is None:
        current_certificates = self.get_current_certificates()
        for certificate in current_certificates:
            if current_certificates[certificate]['alias'] == 'jetty':
                self.cache_is_public_server_certificate_signed = current_certificates[certificate]['type'] == 'caSigned'
                break
    return self.cache_is_public_server_certificate_signed