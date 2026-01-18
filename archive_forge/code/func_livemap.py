from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
@cached_property
def livemap(self):
    return analysis.compute_live_map(self.cfg, self._blocks, self.usedefs.usemap, self.usedefs.defmap)