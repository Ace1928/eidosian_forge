from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def get_overwatch_client():
    return apis.GetClientInstance('securedlandingzone', 'v1beta', no_http=False)