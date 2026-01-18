import ast
import sys
from functools import wraps
from typing import Callable, List, TypeVar
import requests
from wandb_gql.client import RetryError
from wandb import env
from wandb.errors import CommError, Error
def parse_backend_error_messages(response: requests.Response) -> List[str]:
    errors = []
    try:
        data = response.json()
    except ValueError:
        return errors
    if 'errors' in data and isinstance(data['errors'], list):
        for error in data['errors']:
            if isinstance(error, str):
                error = {'message': error}
            if 'message' in error:
                errors.append(error['message'])
    return errors