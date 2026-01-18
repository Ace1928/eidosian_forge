import abc
import stevedore
from keystoneauth1 import exceptions
def load_from_options_getter(self, getter, **kwargs):
    """Load a plugin from getter function that returns appropriate values.

        To handle cases other than the provided CONF and CLI loading you can
        specify a custom loader function that will be queried for the option
        value.
        The getter is a function that takes a
        :py:class:`keystoneauth1.loading.Opt` and returns a value to load with.

        :param getter: A function that returns a value for the given opt.
        :type getter: callable

        :returns: An authentication Plugin.
        :rtype: :py:class:`keystoneauth1.plugin.BaseAuthPlugin`
        """
    for opt in (o for o in self.get_options() if o.dest not in kwargs):
        val = getter(opt)
        if val is not None:
            val = opt.type(val)
        kwargs[opt.dest] = val
    return self.load_from_options(**kwargs)