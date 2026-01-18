import argparse
from cliff import show
from vitrageclient.common import utils
from vitrageclient import exceptions as exc
@staticmethod
def positive_non_zero_int(argument_value):
    if argument_value is None:
        return None
    try:
        value = int(argument_value)
    except ValueError:
        msg = '%s must be an integer' % argument_value
        raise argparse.ArgumentTypeError(msg)
    if value <= 0:
        msg = '%s must be greater than 0' % argument_value
        raise argparse.ArgumentTypeError(msg)
    return value