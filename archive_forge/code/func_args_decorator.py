import datetime
import errno
import functools
import hashlib
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
import wrapt
from glanceclient._i18n import _
from glanceclient import exc
def args_decorator(func):

    def prepare_fields(fields):
        args = ('--' + x.replace('_', '-') for x in fields)
        return ', '.join(args)

    @functools.wraps(func)
    def func_wrapper(gc, args):
        fields = set((a[0] for a in vars(args).items() if a[1]))
        present = fields.intersection(data_fields)
        missing = set(required) - fields
        if (present or get_data_file(args)) and missing:
            msg = _('error: Must provide %(req)s when using %(opt)s.') % {'req': prepare_fields(missing), 'opt': prepare_fields(present) or 'stdin'}
            raise exc.CommandError(msg)
        return func(gc, args)
    return func_wrapper