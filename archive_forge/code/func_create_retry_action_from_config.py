import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def create_retry_action_from_config(config, operation_name=None):
    delay_config = config['__default__']['delay']
    if delay_config['type'] == 'exponential':
        return create_exponential_delay_function(base=delay_config['base'], growth_factor=delay_config['growth_factor'])