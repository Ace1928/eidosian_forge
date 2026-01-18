import itertools
import os
from keystoneauth1.loading import _utils
@property
def argparse_default(self):
    for o in self._all_opts:
        v = os.environ.get('OS_%s' % o.name.replace('-', '_').upper())
        if v:
            return v
    return self.default