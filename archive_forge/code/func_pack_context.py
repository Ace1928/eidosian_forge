import collections
import uuid
from oslo_config import cfg
from oslo_messaging._drivers import common as rpc_common
def pack_context(msg, context):
    """Pack context into msg.

    Values for message keys need to be less than 255 chars, so we pull
    context out into a bunch of separate keys. If we want to support
    more arguments in rabbit messages, we may want to do the same
    for args at some point.

    """
    if isinstance(context, dict):
        context_d = context.items()
    else:
        context_d = context.to_dict().items()
    msg.update((('_context_%s' % key, value) for key, value in context_d))