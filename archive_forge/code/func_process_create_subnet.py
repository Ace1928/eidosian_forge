import abc
from neutron_lib.api.definitions import portbindings
def process_create_subnet(self, plugin_context, data, result):
    """Process extended attributes for create subnet.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming subnet data
        :param result: subnet dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and persist any extended subnet attributes defined by this
        driver. Extended attribute values must also be added to
        result.
        """
    pass