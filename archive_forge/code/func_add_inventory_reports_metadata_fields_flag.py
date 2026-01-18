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
def add_inventory_reports_metadata_fields_flag(parser, require_create_flags=False):
    """Adds the metadata-fields flag."""
    parser.add_argument('--metadata-fields', metavar='METADATA_FIELDS', default=list(ALL_INVENTORY_REPORTS_METADATA_FIELDS) if require_create_flags else None, type=ArgListWithRequiredFieldsCheck(choices=ALL_INVENTORY_REPORTS_METADATA_FIELDS), help='The metadata fields to be included in the inventory report. The fields: "{}" are REQUIRED. '.format(', '.join(REQUIRED_INVENTORY_REPORTS_METADATA_FIELDS)) + _get_optional_help_text(require_create_flags, 'metadata_fields'))