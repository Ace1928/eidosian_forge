import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
@staticmethod
def package(name, path):
    """Loader factory for loading templates from egg package data.
        
        :param name: the name of the package containing the resources
        :param path: the path inside the package data
        :return: the loader function to load templates from the given package
        :rtype: ``function``
        """
    from pkg_resources import resource_stream

    def _load_from_package(filename):
        filepath = os.path.join(path, filename)
        return (filepath, filename, resource_stream(name, filepath), None)
    return _load_from_package