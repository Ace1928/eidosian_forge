from __future__ import absolute_import, division, print_function
from . import lexer, error
from . import coretypes
def parse_boolean_list(self):
    """
        boolean_list : boolean COMMA boolean_list
                     | boolean

        Returns a list of booleans, or None.
        """
    return self.parse_homogeneous_list(self.parse_boolean, lexer.COMMA, 'Expected another boolean, ' + 'type constructor parameter ' + 'lists must have uniform type')