from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def producer_service_flag(suffix='to act on', flag_name='service'):
    return base.Argument(flag_name, completer=ProducerServiceCompleter, help='The name of the service {0}.'.format(suffix))