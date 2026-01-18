import os
import warnings
from enchant.errors import Error, DictNotFoundError
from enchant.utils import get_default_language
from enchant.pypwl import PyPWL
def request_pwl_dict(self, pwl):
    """Request a Dict object for a personal word list.

        This method behaves as 'request_dict' but rather than returning
        a dictionary for a specific language, it returns a dictionary
        referencing a personal word list.  A personal word list is a file
        of custom dictionary entries, one word per line.
        """
    self._check_this()
    new_dict = _e.broker_request_pwl_dict(self._this, pwl.encode())
    if new_dict is None:
        e_str = "Personal Word List file '%s' could not be loaded"
        self._raise_error(e_str % (pwl,))
    if new_dict not in self._live_dicts:
        self._live_dicts[new_dict] = 1
    else:
        self._live_dicts[new_dict] += 1
    d = Dict(False)
    d._switch_this(new_dict, self)
    return d