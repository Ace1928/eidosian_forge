import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@_Cache.me
def cc_test_cexpr(self, cexpr, flags=[]):
    """
        Same as the above but supports compile-time expressions.
        """
    self.dist_log('testing compiler expression', cexpr)
    test_path = os.path.join(self.conf_tmp_path, 'npy_dist_test_cexpr.c')
    with open(test_path, 'w') as fd:
        fd.write(textwrap.dedent(f'               #if !({cexpr})\n                   #error "unsupported expression"\n               #endif\n               int dummy;\n            '))
    test = self.dist_test(test_path, flags)
    if not test:
        self.dist_log('testing failed', stderr=True)
    return test