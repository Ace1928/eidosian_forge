import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def create_retry_handler(config, operation_name=None):
    checker = create_checker_from_retry_config(config, operation_name=operation_name)
    action = create_retry_action_from_config(config, operation_name=operation_name)
    return RetryHandler(checker=checker, action=action)