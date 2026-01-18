import decimal
import json
from datetime import datetime
from botocore.exceptions import ParamValidationError
from botocore.utils import is_json_value_header, parse_to_aware_datetime
def generate_report(self):
    error_messages = []
    for error in self._errors:
        error_messages.append(self._format_error(error))
    return '\n'.join(error_messages)