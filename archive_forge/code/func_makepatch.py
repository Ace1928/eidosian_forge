from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
def makepatch(original, modified):
    """Create a patch object.

  Some methods support PATCH, an efficient way to send updates to a resource.
  This method allows the easy construction of patch bodies by looking at the
  differences between a resource before and after it was modified.

  Args:
    original: object, the original deserialized resource
    modified: object, the modified deserialized resource
  Returns:
    An object that contains only the changes from original to modified, in a
    form suitable to pass to a PATCH method.

  Example usage:
    item = service.activities().get(postid=postid, userid=userid).execute()
    original = copy.deepcopy(item)
    item['object']['content'] = 'This is updated.'
    service.activities.patch(postid=postid, userid=userid,
      body=makepatch(original, item)).execute()
  """
    patch = {}
    for key, original_value in six.iteritems(original):
        modified_value = modified.get(key, None)
        if modified_value is None:
            patch[key] = None
        elif original_value != modified_value:
            if type(original_value) == type({}):
                patch[key] = makepatch(original_value, modified_value)
            else:
                patch[key] = modified_value
        else:
            pass
    for key in modified:
        if key not in original:
            patch[key] = modified[key]
    return patch