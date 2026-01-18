from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def make_request_model(name, module, body_params: List[ModelField]):
    whole_params_list = [p for p in body_params if isinstance(p.field_info, Params)]
    if len(whole_params_list):
        if len(whole_params_list) > 1:
            raise RuntimeError(f"Only one 'Params' allowed: params={whole_params_list}")
        body_params_list = [p for p in body_params if not isinstance(p.field_info, Params)]
        if body_params_list:
            raise RuntimeError(f"No other params allowed when 'Params' used: params={whole_params_list}, other={body_params_list}")
    if whole_params_list:
        assert whole_params_list[0].alias == 'params'
        params_field = whole_params_list[0]
    else:
        _JsonRpcRequestParams = ModelMetaclass.__new__(ModelMetaclass, '_JsonRpcRequestParams', (BaseModel,), {})
        for f in body_params:
            _JsonRpcRequestParams.__fields__[f.name] = f
        _JsonRpcRequestParams = component_name(f'_Params[{name}]', module)(_JsonRpcRequestParams)
        params_field = ModelField(name='params', type_=_JsonRpcRequestParams, class_validators={}, default=None, required=True, model_config=BaseConfig, field_info=Field(...))

    class _Request(BaseModel):
        jsonrpc: StrictStr = Field('2.0', const=True, example='2.0')
        id: Union[StrictStr, int] = Field(None, example=0)
        method: StrictStr = Field(name, const=True, example=name)

        class Config:
            extra = 'forbid'
    _Request.__fields__[params_field.name] = params_field
    _Request = component_name(f'_Request[{name}]', module)(_Request)
    return _Request