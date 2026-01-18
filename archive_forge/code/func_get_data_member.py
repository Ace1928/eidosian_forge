import re
import jmespath
from botocore import xform_name
from ..exceptions import ResourceLoadException
def get_data_member(parent, path):
    """
    Get a data member from a parent using a JMESPath search query,
    loading the parent if required. If the parent cannot be loaded
    and no data is present then an exception is raised.

    :type parent: ServiceResource
    :param parent: The resource instance to which contains data we
                   are interested in.
    :type path: string
    :param path: The JMESPath expression to query
    :raises ResourceLoadException: When no data is present and the
                                   resource cannot be loaded.
    :returns: The queried data or ``None``.
    """
    if parent.meta.data is None:
        if hasattr(parent, 'load'):
            parent.load()
        else:
            raise ResourceLoadException('{0} has no load method!'.format(parent.__class__.__name__))
    return jmespath.search(path, parent.meta.data)