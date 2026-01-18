from contextlib import contextmanager
import copy
import datetime
import io
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from unittest import mock
from dateutil import tz
from oslo_config import cfg
from oslo_config import fixture as fixture_config  # noqa
from oslo_context import context
from oslo_context import fixture as fixture_context
from oslo_i18n import fixture as fixture_trans
from oslo_serialization import jsonutils
from oslotest import base as test_base
import testtools
from oslo_log import _options
from oslo_log import formatters
from oslo_log import handlers
from oslo_log import log
from oslo_utils import units
def mk_log_config(self, data):
    """Turns a dictConfig-like structure into one suitable for fileConfig.

        The schema is not validated as this is a test helper not production
        code. Garbage in, garbage out. Particularly, don't try to use filters,
        fileConfig doesn't support them.

        Handler args must be passed like 'args': (1, 2). dictConfig passes
        keys by keyword name and fileConfig passes them by position so
        accepting the dictConfig form makes it nigh impossible to produce the
        fileConfig form.

        I traverse dicts by sorted keys for output stability but it doesn't
        matter if defaulted keys are out of order.
        """
    lines = []
    for section in ['formatters', 'handlers', 'loggers']:
        items = data.get(section, {})
        keys = sorted(items)
        skeys = ','.join(keys)
        if section == 'loggers' and 'root' in data:
            skeys = 'root,' + skeys if skeys else 'root'
        lines.extend(['[%s]' % section, 'keys=%s' % skeys])
        for key in keys:
            lines.extend(['', '[%s_%s]' % (section[:-1], key)])
            item = items[key]
            lines.extend(('%s=%s' % (k, item[k]) for k in sorted(item)))
            if section == 'handlers':
                if 'args' not in item:
                    lines.append('args=()')
            elif section == 'loggers':
                lines.append('qualname=%s' % key)
                if 'handlers' not in item:
                    lines.append('handlers=')
        lines.append('')
    root = data.get('root', {})
    if root:
        lines.extend(['[logger_root]'])
        lines.extend(('%s=%s' % (k, root[k]) for k in sorted(root)))
        if 'handlers' not in root:
            lines.append('handlers=')
    return '\n'.join(lines)