import builtins as builtin_mod
import sys
import io as _io
import tokenize
from traitlets.config.configurable import Configurable
from traitlets import Instance, Float
from warnings import warn
def cull_cache(self):
    """Output cache is full, cull the oldest entries"""
    oh = self.shell.user_ns.get('_oh', {})
    sz = len(oh)
    cull_count = max(int(sz * self.cull_fraction), 2)
    warn('Output cache limit (currently {sz} entries) hit.\nFlushing oldest {cull_count} entries.'.format(sz=sz, cull_count=cull_count))
    for i, n in enumerate(sorted(oh)):
        if i >= cull_count:
            break
        self.shell.user_ns.pop('_%i' % n, None)
        oh.pop(n, None)