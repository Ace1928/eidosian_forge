from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
@classmethod
def build_resp_model(cls):
    ns = {'code': Field(cls.CODE, const=True, example=cls.CODE), 'message': Field(cls.MESSAGE, const=True, example=cls.MESSAGE), '__annotations__': {'code': int, 'message': str}}
    data_model = cls.get_data_model()
    if data_model is not None:
        if not cls.data_required:
            data_model = Optional[data_model]
        ns['__annotations__']['data'] = data_model
    name = cls._component_name or cls.__name__
    _JsonRpcErrorModel = ModelMetaclass.__new__(ModelMetaclass, '_JsonRpcErrorModel', (BaseModel,), ns)
    _JsonRpcErrorModel = component_name(name, cls.__module__)(_JsonRpcErrorModel)

    @component_name(f'_ErrorResponse[{name}]', cls.__module__)
    class _ErrorResponseModel(BaseModel):
        jsonrpc: StrictStr = Field('2.0', const=True, example='2.0')
        id: Union[StrictStr, int] = Field(None, example=0)
        error: _JsonRpcErrorModel

        class Config:
            extra = 'forbid'
    return _ErrorResponseModel