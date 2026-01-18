from collections import OrderedDict
from ... import sage_helper
@cached_method
def cochain_complex(self):
    return self.chain_complex().dual()