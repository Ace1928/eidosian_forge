from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def document_shape_type_map(self, section, shape, history, include=None, exclude=None, **kwargs):
    self._add_member_documentation(section, shape, **kwargs)
    key_section = section.add_new_section('key', context={'shape': shape.key.name})
    self._start_nested_param(key_section)
    self._add_member_documentation(key_section, shape.key)
    param_section = section.add_new_section(shape.value.name, context={'shape': shape.value.name})
    param_section.style.indent()
    self._start_nested_param(param_section)
    self.traverse_and_document_shape(section=param_section, shape=shape.value, history=history, name=None)
    end_section = section.add_new_section('end-map')
    self._end_nested_param(end_section)
    self._end_nested_param(end_section)