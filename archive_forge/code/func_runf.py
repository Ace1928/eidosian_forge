import sys
from io import StringIO
from unittest import TestCase
from IPython.testing import tools as tt
from IPython.core.magic import (
def runf():
    """Marker function: sets a flag when executed.
        """
    ip.user_ns['code_ran'] = True
    return 'runf'