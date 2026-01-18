from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
@classmethod
def build_data_model(cls):
    if cls.DataModel is not None:
        return rename_if_scope_child_component(cls, cls.DataModel, 'Data')
    error_model = cls.get_error_model()
    if error_model is None:
        return None
    errors_annotation = List[error_model]
    if not cls.errors_required:
        errors_annotation = Optional[errors_annotation]
    ns = {'__annotations__': {'errors': errors_annotation}}
    _ErrorData = ModelMetaclass.__new__(ModelMetaclass, '_ErrorData', (BaseModel,), ns)
    _ErrorData = component_name(f'_ErrorData[{error_model.__name__}]', error_model.__module__)(_ErrorData)
    return _ErrorData