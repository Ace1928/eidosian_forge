import threading
from paramiko import util
from paramiko.common import (
def check_channel_env_request(self, channel, name, value):
    """
        Check whether a given environment variable can be specified for the
        given channel.  This method should return ``True`` if the server
        is willing to set the specified environment variable.  Note that
        some environment variables (e.g., PATH) can be exceedingly
        dangerous, so blindly allowing the client to set the environment
        is almost certainly not a good idea.

        The default implementation always returns ``False``.

        :param channel: the `.Channel` the env request arrived on
        :param str name: name
        :param str value: Channel value
        :returns: A boolean
        """
    return False