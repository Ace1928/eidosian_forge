import logging
import random
from botocore.exceptions import (
from botocore.retries import quota, special
from botocore.retries.base import BaseRetryableChecker, BaseRetryBackoff
def detect_error_type(self, context):
    """Detect the error type associated with an error code and model.

        This will either return:

            * ``self.TRANSIENT_ERROR`` - If the error is a transient error
            * ``self.THROTTLING_ERROR`` - If the error is a throttling error
            * ``None`` - If the error is neither type of error.

        """
    error_code = context.get_error_code()
    op_model = context.operation_model
    if op_model is None or not op_model.error_shapes:
        return
    for shape in op_model.error_shapes:
        if shape.metadata.get('retryable') is not None:
            error_code_to_check = shape.metadata.get('error', {}).get('code') or shape.name
            if error_code == error_code_to_check:
                if shape.metadata['retryable'].get('throttling'):
                    return self.THROTTLING_ERROR
                return self.TRANSIENT_ERROR