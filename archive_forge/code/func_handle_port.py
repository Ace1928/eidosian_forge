import abc
from neutron_lib.agent import extension
@abc.abstractmethod
def handle_port(self, context, data):
    """Handle a port add/update event.

        This can be called on either create or update, depending on the
        code flow. Thus, it's this function's responsibility to check what
        actually changed.

        :param context: RPC context.
        :param data: Port data.
        """