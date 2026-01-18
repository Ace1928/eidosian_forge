from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
def invalid_request_from_validation_error(exc: ValidationError) -> InvalidRequest:
    return InvalidRequest(data={'errors': exc.errors()})