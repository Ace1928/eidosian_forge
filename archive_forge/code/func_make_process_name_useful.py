from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
def make_process_name_useful():
    """Sets the process name to something better than 'python' if possible."""
    set_kernel_process_name(os.path.basename(sys.argv[0]))