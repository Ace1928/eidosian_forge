import collections
import itertools
import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import plugin_loader
def load_from_module(self, module):
    """Return the mapping specified in the given module.

        If no such mapping is specified, an empty dictionary is returned.
        """
    for mapping_name in self.names:
        mapping_func = getattr(module, mapping_name, None)
        if callable(mapping_func):
            fmt_data = {'mapping_name': mapping_name, 'module': module}
            try:
                mapping_dict = mapping_func(*self.args, **self.kwargs)
            except Exception:
                LOG.error('Failed to load %(mapping_name)s from %(module)s', fmt_data)
                raise
            else:
                if isinstance(mapping_dict, collections.abc.Mapping):
                    return mapping_dict
                elif mapping_dict is not None:
                    LOG.error('Invalid type for %(mapping_name)s from %(module)s', fmt_data)
    return {}