from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import k8s_object
@property
def task_template(self):
    return self.template