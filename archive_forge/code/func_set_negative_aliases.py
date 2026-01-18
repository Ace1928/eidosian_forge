import sys, string, re
import getopt
from distutils.errors import *
def set_negative_aliases(self, negative_alias):
    """Set the negative aliases for this option parser.
        'negative_alias' should be a dictionary mapping option names to
        option names, both the key and value must already be defined
        in the option table."""
    self._check_alias_dict(negative_alias, 'negative alias')
    self.negative_alias = negative_alias