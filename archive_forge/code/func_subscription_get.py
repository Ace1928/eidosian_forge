import datetime
import json
from oslo_utils import timeutils
from zaqarclient.queues.v1 import core
def subscription_get(transport, request, queue_name, subscription_id):
    """Gets a particular subscription data

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param subscription_id: ID of subscription.
    :type subscription_id: str

    """
    request.operation = 'subscription_get'
    request.params['queue_name'] = queue_name
    request.params['subscription_id'] = subscription_id
    resp = transport.send(request)
    return resp.deserialized_content