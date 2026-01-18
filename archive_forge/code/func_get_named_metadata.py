import collections
from llvmlite.ir import context, values, types, _utils
def get_named_metadata(self, name):
    """
        Return the metadata node with the given *name*.  KeyError is raised
        if no such node exists (contrast with add_named_metadata()).
        """
    return self.namedmetadata[name]