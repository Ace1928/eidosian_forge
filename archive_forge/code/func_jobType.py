from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import util
@property
def jobType(self):
    return self.getTruncatedFieldNameBySuffix('Job')