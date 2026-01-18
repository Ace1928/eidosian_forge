from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
@cached_property
def deadmaps(self):
    return analysis.compute_dead_maps(self.cfg, self._blocks, self.livemap, self.usedefs.defmap)