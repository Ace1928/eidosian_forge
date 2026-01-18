from __future__ import absolute_import, division, print_function
import re
from .common import F5ModuleError
def build_service_uri(base_uri, partition, name):
    """Build the proper uri for a service resource.
    This follows the scheme:
        <base_uri>/~<partition>~<<name>.app>~<name>
    :param base_uri: str -- base uri of the REST endpoint
    :param partition: str -- partition for the service
    :param name: str -- name of the service
    :returns: str -- uri to access the service
    """
    name = name.replace('/', '~')
    return '%s~%s~%s.app~%s' % (base_uri, partition, name, name)