from ...error import GraphQLError
from ...language.printer import print_ast
from ...type.definition import GraphQLNonNull
from ...utils.is_valid_literal_value import is_valid_literal_value
from .base import ValidationRule
@staticmethod
def bad_value_for_default_arg_message(var_name, type, value, verbose_errors):
    message = u'\n' + u'\n'.join(verbose_errors) if verbose_errors else u''
    return u'Variable "${}" of type "{}" has invalid default value: {}.{}'.format(var_name, type, value, message)