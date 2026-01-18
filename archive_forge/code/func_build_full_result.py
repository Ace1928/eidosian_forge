import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def build_full_result(self):
    complete_result = {}
    for response in self:
        page = response
        if isinstance(response, tuple) and len(response) == 2:
            page = response[1]
        for result_expression in self.result_keys:
            result_value = result_expression.search(page)
            if result_value is None:
                continue
            existing_value = result_expression.search(complete_result)
            if existing_value is None:
                set_value_from_jmespath(complete_result, result_expression.expression, result_value)
                continue
            if isinstance(result_value, list):
                existing_value.extend(result_value)
            elif isinstance(result_value, (int, float, str)):
                set_value_from_jmespath(complete_result, result_expression.expression, existing_value + result_value)
    merge_dicts(complete_result, self.non_aggregate_part)
    if self.resume_token is not None:
        complete_result['NextToken'] = self.resume_token
    return complete_result