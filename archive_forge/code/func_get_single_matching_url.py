from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks import task_util
def get_single_matching_url(url_string):
    """Gets cloud resource, allowing wildcards that match only one resource."""
    if not wildcard_iterator.contains_wildcard(url_string):
        return storage_url.storage_url_from_string(url_string)
    resource_iterator = wildcard_iterator.get_wildcard_iterator(url_string, fields_scope=cloud_api.FieldsScope.SHORT)
    plurality_checkable_resource_iterator = plurality_checkable_iterator.PluralityCheckableIterator(resource_iterator)
    if plurality_checkable_resource_iterator.is_plural():
        raise errors.InvalidUrlError('get-iam-policy must match a single cloud resource.')
    return list(plurality_checkable_resource_iterator)[0].storage_url