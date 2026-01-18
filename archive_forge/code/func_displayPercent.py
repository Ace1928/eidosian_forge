from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
@property
def displayPercent(self):
    """Returns human readable revision percent."""
    if self.statusPercent == self.specPercent:
        return _FormatPercentage(self.statusPercent)
    else:
        return '{:4} (currently {})'.format(_FormatPercentage(self.specPercent), _FormatPercentage(self.statusPercent))