from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import portbindings_extended
from neutron_lib.api.definitions import subnet
def get_topic_name(prefix, table, operation, host=None):
    """Create a topic name.

    The topic name needs to be synced between the agent and the
    plugin. The plugin will send a fanout message to all of the
    listening agents so that the agents in turn can perform their
    updates accordingly.

    :param prefix: Common prefix for the plugin/agent message queues.
    :param table: The table in question (NETWORK, SUBNET, PORT).
    :param operation: The operation that invokes notification (CREATE,
                      DELETE, UPDATE)
    :param host: Add host to the topic
    :returns: The topic name.
    """
    if host:
        return '%s-%s-%s.%s' % (prefix, table, operation, host)
    return '%s-%s-%s' % (prefix, table, operation)