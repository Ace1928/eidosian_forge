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
def add_key_update_args(parser):
    """Adds args for api-keys update command."""
    update_set_restriction_group = parser.add_mutually_exclusive_group(required=False)
    _add_clear_restrictions_arg(update_set_restriction_group)
    restriction_group = update_set_restriction_group.add_argument_group()
    client_restriction_group = restriction_group.add_mutually_exclusive_group()
    _allowed_referrers_arg(client_restriction_group)
    _allowed_ips_arg(client_restriction_group)
    _allowed_bundle_ids(client_restriction_group)
    _allowed_application(client_restriction_group)
    _api_targets_arg(restriction_group)
    update_set_annotation_group = parser.add_mutually_exclusive_group(required=False)
    _annotations(update_set_annotation_group)
    _add_clear_annotations_arg(update_set_annotation_group)