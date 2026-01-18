from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
@property
def resource_argument_collection(self):
    """Gets the collection that should be used to represent the resource.

    Most of the time this is the same as request_collection because all methods
    in a collection operate on the same resource and so the API method takes
    the same parameters that make up the resource.

    One exception is List methods where the API parameters are for the parent
    collection. Because people don't specify the resource directly for list
    commands this also returns the parent collection for parsing purposes.

    The other exception is Create methods. They reference the parent collection
    list Like, but the difference is that we *do* want to specify the actual
    resource on the command line, so the original resource collection is
    returned here instead of the one that matches the API methods. When
    generating the request, you must figure out how to generate the message
    correctly from the parsed resource (as you cannot simply pass the reference
    to the API).

    Returns:
      APICollection: The collection.
    """
    if self.IsList():
        return self._request_collection
    return self.collection