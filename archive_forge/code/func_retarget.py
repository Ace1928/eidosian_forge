import abc
import weakref
from numba.core import errors
def retarget(self, orig_disp):
    """Apply retargeting to orig_disp.

        The retargeted dispatchers are cached for future use.
        """
    cache = self.cache
    opts = orig_disp.targetoptions
    if opts.get('target_backend') == self.output_target:
        return orig_disp
    cached = cache.load_cache(orig_disp)
    if cached is None:
        out = self.compile_retarget(orig_disp)
        cache.save_cache(orig_disp, out)
    else:
        out = cached
    return out