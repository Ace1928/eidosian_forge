import argparse
import functools
import hashlib
import logging
import os
import socket
import time
import urllib.parse
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
import requests
from keystoneclient import exceptions
from keystoneclient.i18n import _
@classmethod
def load_from_cli_options(cls, args, **kwargs):
    """Create a :py:class:`.Session` object from CLI arguments.

        The CLI arguments must have been registered with
        :py:meth:`.register_cli_options`.

        :param Namespace args: result of parsed arguments.

        :returns: A new session object.
        :rtype: :py:class:`.Session`
        """
    kwargs['insecure'] = args.insecure
    kwargs['cacert'] = args.os_cacert
    if args.os_cert and args.os_key:
        kwargs['cert'] = (args.os_cert, args.os_key)
    kwargs['timeout'] = args.timeout
    return cls._make(**kwargs)