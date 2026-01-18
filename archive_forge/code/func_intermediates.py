import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
@intermediates.setter
@_immutable_after_save
def intermediates(self, value):
    super(CertificateContainer, self).remove('intermediates')
    super(CertificateContainer, self).add('intermediates', value)