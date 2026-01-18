import abc
from neutron_lib.api.definitions import portbindings
def process_create_network(self, plugin_context, data, result):
    """Process extended attributes for create network.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming network data
        :param result: network dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and persist any extended network attributes defined by this
        driver. Extended attribute values must also be added to
        result.
        """
    pass