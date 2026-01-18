from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
@property
def displayTags(self):
    spec_tags = self.specTags
    status_tags = self.statusTags
    if spec_tags == status_tags:
        return status_tags if status_tags != _MISSING_PERCENT_OR_TAGS else ''
    else:
        return '{} (currently {})'.format(spec_tags, status_tags)