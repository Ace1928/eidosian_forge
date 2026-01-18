from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import messaging
from heat.rpc import api as rpc_api
def ignore_error_by_name(self, name):
    """Returns a context manager that filters exceptions with a given name.

        :param name: Name to compare the local exception name to.
        """

    def error_name_matches(err):
        return self.local_error_name(err) == name
    return excutils.exception_filter(error_name_matches)