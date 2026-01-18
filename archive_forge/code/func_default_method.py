from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
@property
def default_method(self):
    """Returns the default API method name for this type of command."""
    return _DEFAULT_METHODS_BY_COMMAND_TYPE.get(self)