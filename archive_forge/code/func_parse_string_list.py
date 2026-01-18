from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_string_list(self):
    """
        string_list : STRING COMMA string_list
                    | STRING

        Returns a list of strings, or None.
        """
    return self.parse_homogeneous_list(self.parse_string, lexer.COMMA, 'Expected another string, ' + 'type constructor parameter ' + 'lists must have uniform type')