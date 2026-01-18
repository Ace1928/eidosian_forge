from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import resources
@property
def clear_arg(self):
    return self._CreateFlag(self.arg_name, flag_prefix=update_args.Prefix.CLEAR, action='store_true', help_text='Clear {} value and set to {}.'.format(self.arg_name, self._GetTextFormatOfEmptyValue(self._empty_value)))