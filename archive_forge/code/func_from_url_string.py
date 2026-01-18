from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
@classmethod
def from_url_string(cls, url_string):
    """Parses the url string and return the storage URL object.

    Args:
      url_string (str): Azure storage URL of the form:
        http://account.blob.core.windows.net/container/blob

    Returns:
      AzureUrl object

    Raises:
      InvalidUrlError: Raised if the url_string is not a valid cloud URL.
    """
    scheme = _get_scheme_from_url_string(url_string)
    AzureUrl.validate_url_string(url_string, scheme)
    schemeless_url_string = url_string[len(scheme.value + SCHEME_DELIMITER):]
    hostname, _, path_and_params = schemeless_url_string.partition(CLOUD_URL_DELIMITER)
    account, _, _ = hostname.partition('.')
    container, _, blob_and_params = path_and_params.partition(CLOUD_URL_DELIMITER)
    blob, _, params = blob_and_params.partition('?')
    params_dict = urllib.parse.parse_qs(params)
    return cls(scheme, bucket_name=container, object_name=blob, generation=params_dict['versionId'][0] if 'versionId' in params_dict else None, snapshot=params_dict['snapshot'][0] if 'snapshot' in params_dict else None, account=account)