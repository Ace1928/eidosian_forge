import collections
import uuid
from oslo_config import cfg
from oslo_messaging._drivers import common as rpc_common
def check_duplicate_message(self, message_data):
    """AMQP consumers may read same message twice when exceptions occur
           before ack is returned. This method prevents doing it.
        """
    try:
        msg_id = message_data.pop(UNIQUE_ID)
    except KeyError:
        return
    if msg_id in self.prev_msgids:
        raise rpc_common.DuplicateMessageError(msg_id=msg_id)
    return msg_id