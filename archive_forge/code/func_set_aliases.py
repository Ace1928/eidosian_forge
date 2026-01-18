import sys, string, re
import getopt
from distutils.errors import *
def set_aliases(self, alias):
    """Set the aliases for this option parser."""
    self._check_alias_dict(alias, 'alias')
    self.alias = alias