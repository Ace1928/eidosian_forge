import itertools
import os
from keystoneauth1.loading import _utils
@property
def argparse_args(self):
    return ['--os-%s' % o.name for o in self._all_opts]