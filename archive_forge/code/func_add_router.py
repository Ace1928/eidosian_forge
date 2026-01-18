import abc
from neutron_lib.agent import extension
@abc.abstractmethod
def add_router(self, context, data):
    """Handle a router add event.

        Called on router create.

        :param context: RPC context.
        :param data: Router data.
        """