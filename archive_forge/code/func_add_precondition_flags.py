from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_precondition_flags(parser):
    """Add flags indicating a precondition for an operation to happen."""
    preconditions_group = parser.add_group(category='PRECONDITION')
    preconditions_group.add_argument('--if-generation-match', metavar='GENERATION', help='Execute only if the generation matches the generation of the requested object.')
    preconditions_group.add_argument('--if-metageneration-match', metavar='METAGENERATION', help='Execute only if the metageneration matches the metageneration of the requested object.')