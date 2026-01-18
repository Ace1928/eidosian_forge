from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
@property
def mqtt_config_enum(self):
    return self.messages.MqttConfig.MqttEnabledStateValueValuesEnum