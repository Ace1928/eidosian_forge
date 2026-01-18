import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
@resume_token.setter
def resume_token(self, value):
    if not isinstance(value, dict):
        raise ValueError('Bad starting token: %s' % value)
    if 'boto_truncate_amount' in value:
        token_keys = sorted(self._input_token + ['boto_truncate_amount'])
    else:
        token_keys = sorted(self._input_token)
    dict_keys = sorted(value.keys())
    if token_keys == dict_keys:
        self._resume_token = self._token_encoder.encode(value)
    else:
        raise ValueError('Bad starting token: %s' % value)