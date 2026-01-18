import argparse
import functools
import hashlib
import logging
import os
from oslo_utils import encodeutils
from oslo_utils import importutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
def http_log_resp(_logger, resp, body):
    if not _logger.isEnabledFor(logging.DEBUG):
        return
    _logger.debug('RESP: %(code)s %(headers)s %(body)s', {'code': resp.status_code, 'headers': resp.headers, 'body': body})