import datetime
import json
from oslo_utils import timeutils
from zaqarclient.queues.v1 import core
def subscription_update(transport, request, queue_name, subscription_id, subscription_data):
    """Updates the subscription

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param subscription_id: ID of subscription.
    :type subscription_id: str
    :param subscription_data: Subscription's properties, i.e: subscriber,
        ttl, options.
    :type subscription_data: `dict`
    """
    request.operation = 'subscription_update'
    request.params['queue_name'] = queue_name
    request.params['subscription_id'] = subscription_id
    request.content = json.dumps(subscription_data)
    resp = transport.send(request)
    return resp.deserialized_content