import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
def get_cached(self, key, **kwargs):
    """Return a value from the :class:`.Cache` referenced by this
        :class:`.Namespace` object's :class:`.Template`.

        The advantage to this method versus direct access to the
        :class:`.Cache` is that the configuration parameters
        declared in ``<%page>`` take effect here, thereby calling
        up the same configured backend as that configured
        by ``<%page>``.

        """
    return self.cache.get(key, **kwargs)