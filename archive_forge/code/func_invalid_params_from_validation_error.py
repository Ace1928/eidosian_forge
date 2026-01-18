from ._base import *
from .models import component_name, rename_if_scope_child_component
from fastapi import status
def invalid_params_from_validation_error(exc: ValidationError) -> InvalidParams:
    errors = []
    for err in exc.errors():
        if err['loc'][:1] == ('body',):
            err['loc'] = err['loc'][1:]
        else:
            assert err['loc']
            err['loc'] = (f'<{err['loc'][0]}>',) + err['loc'][1:]
        errors.append(err)
    return InvalidParams(data={'errors': errors})