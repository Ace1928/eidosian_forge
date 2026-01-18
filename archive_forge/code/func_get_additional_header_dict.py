from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import properties
def get_additional_header_dict():
    """Gets a dictionary of headers for API calls based on a property value."""
    headers_string = properties.VALUES.storage.additional_headers.Get()
    if not headers_string:
        return {}
    parser = arg_parsers.ArgDict()
    headers_dict = parser(headers_string)
    return _remove_metadata_headers(headers_dict)