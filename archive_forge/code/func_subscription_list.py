import datetime
import json
from oslo_utils import timeutils
from zaqarclient.queues.v1 import core
def subscription_list(transport, request, queue_name, **kwargs):
    """Gets a list of subscriptions

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param kwargs: Optional arguments for this operation.
        - marker: Where to start getting subscriptions from.
        - limit: Maximum number of subscriptions to get.
    """
    request.operation = 'subscription_list'
    request.params['queue_name'] = queue_name
    request.params.update(kwargs)
    resp = transport.send(request)
    if not resp.content:
        return {'links': [], 'subscriptions': []}
    return resp.deserialized_content