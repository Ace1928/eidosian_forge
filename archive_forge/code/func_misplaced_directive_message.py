from ...error import GraphQLError
from ...language import ast
from ...type.directives import DirectiveLocation
from .base import ValidationRule
@staticmethod
def misplaced_directive_message(directive_name, location):
    return 'Directive "{}" may not be used on "{}".'.format(directive_name, location)