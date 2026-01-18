from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
@property
def last_transition_time(self):
    assert hasattr(self, 'READY_CONDITION')
    if self.ready_condition:
        return self.ready_condition['lastTransitionTime']