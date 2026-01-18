from ...error import GraphQLError
from ...language.printer import print_ast
from ...utils.is_valid_literal_value import is_valid_literal_value
from .base import ValidationRule
@staticmethod
def bad_value_message(arg_name, type, value, verbose_errors):
    message = u'\n' + u'\n'.join(verbose_errors) if verbose_errors else ''
    return 'Argument "{}" has invalid value {}.{}'.format(arg_name, value, message)