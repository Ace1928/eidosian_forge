import json
import numbers
from collections import OrderedDict
from functools import update_wrapper
from pprint import pformat
from typing import Any
import click
from click import Context, ParamType
from kombu.utils.objects import cached_property
from celery._state import get_current_app
from celery.signals import user_preload_options
from celery.utils import text
from celery.utils.log import mlevel
from celery.utils.time import maybe_iso8601
def pretty_dict_ok_error(self, n):
    try:
        return (self.OK, text.indent(self.pretty(n['ok'])[1], 4))
    except KeyError:
        pass
    return (self.ERROR, text.indent(self.pretty(n['error'])[1], 4))