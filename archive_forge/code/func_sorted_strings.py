from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def sorted_strings(cidrs):
    return [six.text_type(cidr) for cidr in sorted(cidrs)]