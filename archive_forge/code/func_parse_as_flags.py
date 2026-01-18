import operator
from numba.core import config, utils
from numba.core.targetconfig import TargetConfig, Option
@classmethod
def parse_as_flags(cls, flags, options):
    """Parse target options defined in ``options`` and set ``flags``
        accordingly.

        Parameters
        ----------
        flags : Flags
        options : dict
        """
    opt = cls()
    opt._apply(flags, options)
    opt.finalize(flags, options)
    return flags