import abc
import copy
from http import client as http_client
from urllib import parse as urlparse
from oslo_utils import strutils
from ironicclient.common.apiclient import exceptions
from ironicclient.common.i18n import _
@classmethod
def run_hooks(cls, hook_type, *args, **kwargs):
    """Run all hooks of specified type.

        :param cls: class that registers hooks
        :param hook_type: hook type, e.g., '__pre_parse_args__'
        :param args: args to be passed to every hook function
        :param kwargs: kwargs to be passed to every hook function
        """
    hook_funcs = cls._hooks_map.get(hook_type) or []
    for hook_func in hook_funcs:
        hook_func(*args, **kwargs)