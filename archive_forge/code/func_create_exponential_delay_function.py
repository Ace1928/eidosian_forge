import functools
import logging
import random
from binascii import crc32
from botocore.exceptions import (
def create_exponential_delay_function(base, growth_factor):
    """Create an exponential delay function based on the attempts.

    This is used so that you only have to pass it the attempts
    parameter to calculate the delay.

    """
    return functools.partial(delay_exponential, base=base, growth_factor=growth_factor)