import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
@payload_content_encoding.setter
@immutable_after_save
def payload_content_encoding(self, value):
    LOG.warning('DEPRECATION WARNING: Manually setting the payload_content_encoding can lead to unexpected results.  It will be removed in a future release. See Launchpad Bug #1419166.')
    self._payload_content_encoding = value