import threading
from paramiko import util
from paramiko.common import (
def check_channel_forward_agent_request(self, channel):
    """
        Determine if the client will be provided with an forward agent session.
        If this method returns ``True``, the server will allow SSH Agent
        forwarding.

        The default implementation always returns ``False``.

        :param .Channel channel: the `.Channel` the request arrived on
        :return: ``True`` if the AgentForward was loaded; ``False`` if not

        If ``True`` is returned, the server should create an
        :class:`AgentServerProxy` to access the agent.
        """
    return False