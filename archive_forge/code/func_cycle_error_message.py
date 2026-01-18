from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def cycle_error_message(fragment_name, spread_names):
    via = ' via {}'.format(', '.join(spread_names)) if spread_names else ''
    return 'Cannot spread fragment "{}" within itself{}.'.format(fragment_name, via)