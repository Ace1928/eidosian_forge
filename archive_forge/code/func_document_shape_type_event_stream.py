from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_shape_type_event_stream(self, section, shape, history, **kwargs):
    self.document_shape_type_structure(section, shape, history, **kwargs)