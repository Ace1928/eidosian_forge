from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
def map_remote_error(ex):
    """Map rpc_common.RemoteError exceptions to HeatAPIException subclasses.

    Map rpc_common.RemoteError exceptions returned by the engine
    to HeatAPIException subclasses which can be used to return
    properly formatted AWS error responses.
    """
    inval_param_errors = ('AttributeError', 'ValueError', 'InvalidTenant', 'EntityNotFound', 'ResourceActionNotSupported', 'ResourceNotFound', 'ResourceNotAvailable', 'StackValidationFailed', 'InvalidSchemaError', 'InvalidTemplateReference', 'InvalidTemplateVersion', 'InvalidTemplateSection', 'UnknownUserParameter', 'UserParameterMissing', 'MissingCredentialError', 'ResourcePropertyConflict', 'PropertyUnspecifiedError', 'NotSupported', 'InvalidBreakPointHook', 'PhysicalResourceIDAmbiguity')
    denied_errors = ('Forbidden', 'NotAuthorized')
    already_exists_errors = 'StackExists'
    invalid_action_errors = ('ActionInProgress',)
    request_limit_exceeded = 'RequestLimitExceeded'
    ex_type = reflection.get_class_name(ex, fully_qualified=False)
    if ex_type.endswith('_Remote'):
        ex_type = ex_type[:-len('_Remote')]
    safe = getattr(ex, 'safe', False)
    detail = str(ex) if safe else None
    if ex_type in inval_param_errors:
        return HeatInvalidParameterValueError(detail=detail)
    elif ex_type in denied_errors:
        return HeatAccessDeniedError(detail=detail)
    elif ex_type in already_exists_errors:
        return AlreadyExistsError(detail=detail)
    elif ex_type in invalid_action_errors:
        return HeatActionInProgressError(detail=detail)
    elif ex_type in request_limit_exceeded:
        return HeatRequestLimitExceeded(detail=detail)
    else:
        return HeatInternalFailureError(detail=detail)