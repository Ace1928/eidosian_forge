from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import services_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def add_key_undelete_args(parser):
    """Adds args for api-keys undelete command."""
    undelete_set_group = parser.add_mutually_exclusive_group(required=True)
    key_flag(undelete_set_group, suffix='to undelete', api_version='v2', required=False)
    _key_string_flag(undelete_set_group)