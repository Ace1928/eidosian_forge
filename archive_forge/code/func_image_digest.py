from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
@property
def image_digest(self):
    """The URL of the image, by digest. Stable when tags are not."""
    return self.status.imageDigest