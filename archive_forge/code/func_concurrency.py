from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
@property
def concurrency(self):
    return self.template.concurrency