import abc
from neutron_lib.api.definitions import portbindings
def process_update_network(self, plugin_context, data, result):
    """Process extended attributes for update network.

        :param plugin_context: plugin request context
        :param data: dictionary of incoming network data
        :param result: network dictionary to extend

        Called inside transaction context on plugin_context.session to
        validate and update any extended network attributes defined by this
        driver. Extended attribute values, whether updated or not,
        must also be added to result.
        """
    pass