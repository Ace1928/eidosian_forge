import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
@staticmethod
def split_filter_param(param):
    key, eq_op, value = param.partition('=')
    if not eq_op:
        msg = 'Malformed parameter(%s). Use the key=value format.' % param
        raise ValueError(msg)
    return (key, value)