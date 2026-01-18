import re
from zaqarclient._i18n import _  # noqa
from zaqarclient import errors
from zaqarclient.queues.v1 import claim as claim_api
from zaqarclient.queues.v1 import core
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
def delete_messages(self, *messages):
    """Deletes a set of messages from the server

        :param messages: List of messages' ids to delete.
        :type messages: *args of str
        """
    req, trans = self.client._request_and_transport()
    return core.message_delete_many(trans, req, self._name, set(messages))