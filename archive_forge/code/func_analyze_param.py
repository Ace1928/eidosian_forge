import inspect
from contextlib import AsyncExitStack, contextmanager
from copy import deepcopy
from typing import (
import anyio
from fastapi import params
from fastapi._compat import (
from fastapi.background import BackgroundTasks
from fastapi.concurrency import (
from fastapi.dependencies.models import Dependant, SecurityRequirement
from fastapi.logger import logger
from fastapi.security.base import SecurityBase
from fastapi.security.oauth2 import OAuth2, SecurityScopes
from fastapi.security.open_id_connect_url import OpenIdConnect
from fastapi.utils import create_response_field, get_path_param_names
from pydantic.fields import FieldInfo
from starlette.background import BackgroundTasks as StarletteBackgroundTasks
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, Headers, QueryParams, UploadFile
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response
from starlette.websockets import WebSocket
from typing_extensions import Annotated, get_args, get_origin
def analyze_param(*, param_name: str, annotation: Any, value: Any, is_path_param: bool) -> Tuple[Any, Optional[params.Depends], Optional[ModelField]]:
    field_info = None
    depends = None
    type_annotation: Any = Any
    use_annotation: Any = Any
    if annotation is not inspect.Signature.empty:
        use_annotation = annotation
        type_annotation = annotation
    if get_origin(use_annotation) is Annotated:
        annotated_args = get_args(annotation)
        type_annotation = annotated_args[0]
        fastapi_annotations = [arg for arg in annotated_args[1:] if isinstance(arg, (FieldInfo, params.Depends))]
        fastapi_specific_annotations = [arg for arg in fastapi_annotations if isinstance(arg, (params.Param, params.Body, params.Depends))]
        if fastapi_specific_annotations:
            fastapi_annotation: Union[FieldInfo, params.Depends, None] = fastapi_specific_annotations[-1]
        else:
            fastapi_annotation = None
        if isinstance(fastapi_annotation, FieldInfo):
            field_info = copy_field_info(field_info=fastapi_annotation, annotation=use_annotation)
            assert field_info.default is Undefined or field_info.default is Required, f'`{field_info.__class__.__name__}` default value cannot be set in `Annotated` for {param_name!r}. Set the default value with `=` instead.'
            if value is not inspect.Signature.empty:
                assert not is_path_param, 'Path parameters cannot have default values'
                field_info.default = value
            else:
                field_info.default = Required
        elif isinstance(fastapi_annotation, params.Depends):
            depends = fastapi_annotation
    if isinstance(value, params.Depends):
        assert depends is None, f'Cannot specify `Depends` in `Annotated` and default value together for {param_name!r}'
        assert field_info is None, f'Cannot specify a FastAPI annotation in `Annotated` and `Depends` as a default value together for {param_name!r}'
        depends = value
    elif isinstance(value, FieldInfo):
        assert field_info is None, f'Cannot specify FastAPI annotations in `Annotated` and default value together for {param_name!r}'
        field_info = value
        if PYDANTIC_V2:
            field_info.annotation = type_annotation
    if depends is not None and depends.dependency is None:
        depends.dependency = type_annotation
    if lenient_issubclass(type_annotation, (Request, WebSocket, HTTPConnection, Response, StarletteBackgroundTasks, SecurityScopes)):
        assert depends is None, f'Cannot specify `Depends` for type {type_annotation!r}'
        assert field_info is None, f'Cannot specify FastAPI annotation for type {type_annotation!r}'
    elif field_info is None and depends is None:
        default_value = value if value is not inspect.Signature.empty else Required
        if is_path_param:
            field_info = params.Path(annotation=use_annotation)
        elif is_uploadfile_or_nonable_uploadfile_annotation(type_annotation) or is_uploadfile_sequence_annotation(type_annotation):
            field_info = params.File(annotation=use_annotation, default=default_value)
        elif not field_annotation_is_scalar(annotation=type_annotation):
            field_info = params.Body(annotation=use_annotation, default=default_value)
        else:
            field_info = params.Query(annotation=use_annotation, default=default_value)
    field = None
    if field_info is not None:
        if is_path_param:
            assert isinstance(field_info, params.Path), f'Cannot use `{field_info.__class__.__name__}` for path param {param_name!r}'
        elif isinstance(field_info, params.Param) and getattr(field_info, 'in_', None) is None:
            field_info.in_ = params.ParamTypes.query
        use_annotation_from_field_info = get_annotation_from_field_info(use_annotation, field_info, param_name)
        if not field_info.alias and getattr(field_info, 'convert_underscores', None):
            alias = param_name.replace('_', '-')
        else:
            alias = field_info.alias or param_name
        field_info.alias = alias
        field = create_response_field(name=param_name, type_=use_annotation_from_field_info, default=field_info.default, alias=alias, required=field_info.default in (Required, Undefined), field_info=field_info)
    return (type_annotation, depends, field)