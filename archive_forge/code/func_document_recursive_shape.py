from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_recursive_shape(self, section, shape, **kwargs):
    self._add_member_documentation(section, shape, **kwargs)