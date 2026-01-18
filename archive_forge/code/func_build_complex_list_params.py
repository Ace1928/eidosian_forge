from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def build_complex_list_params(self, params, items, label, names):
    """Serialize a list of structures.

        For example::

            items = [('foo', 'bar', 'baz'), ('foo2', 'bar2', 'baz2')]
            label = 'ParamName.member'
            names = ('One', 'Two', 'Three')
            self.build_complex_list_params(params, items, label, names)

        would result in the params dict being updated with these params::

            ParamName.member.1.One = foo
            ParamName.member.1.Two = bar
            ParamName.member.1.Three = baz

            ParamName.member.2.One = foo2
            ParamName.member.2.Two = bar2
            ParamName.member.2.Three = baz2

        :type params: dict
        :param params: The params dict.  The complex list params
            will be added to this dict.

        :type items: list of tuples
        :param items: The list to serialize.

        :type label: string
        :param label: The prefix to apply to the parameter.

        :type names: tuple of strings
        :param names: The names associated with each tuple element.

        """
    for i, item in enumerate(items, 1):
        current_prefix = '%s.%s' % (label, i)
        for key, value in zip(names, item):
            full_key = '%s.%s' % (current_prefix, key)
            params[full_key] = value