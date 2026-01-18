import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def find_resource_by_id(client, resource, resource_id, cmd_resource=None, parent_id=None, fields=None):
    return client.find_resource_by_id(resource, resource_id, cmd_resource, parent_id, fields)