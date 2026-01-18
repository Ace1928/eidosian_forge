import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
@contextlib.contextmanager
def check_missing_extension_if_error(client_manager, attrs):
    try:
        yield
    except openstack.exceptions.HttpException:
        for opt, ext in _required_opt_extensions_map.items():
            if opt in attrs:
                client_manager.find_extension(ext, ignore_missing=False)
        raise