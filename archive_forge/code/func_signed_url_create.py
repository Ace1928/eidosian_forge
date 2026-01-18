import datetime
import json
from oslo_utils import timeutils
from zaqarclient.queues.v1 import core
def signed_url_create(transport, request, queue_name, paths=None, ttl_seconds=None, project_id=None, methods=None):
    """Creates a signed URL given a queue name

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: name of Queue for the URL to access
    :type name: str
    :param paths: Allowed actions. Options: messages, subscriptions, claims
    :type name: list
    :param ttl_seconds: Seconds the URL will be valid for, default 86400
    :type name: int
    :param project_id: defaults to None
    :type name: str
    :param methods: HTTP methods to allow, defaults to ["GET"]
    :type name: `list`
    """
    request.operation = 'signed_url_create'
    request.params['queue_name'] = queue_name
    body = {}
    if ttl_seconds is not None:
        expiry = timeutils.utcnow() + datetime.timedelta(seconds=ttl_seconds)
        body['expires'] = expiry.isoformat()
    if project_id is not None:
        body['project_id'] = project_id
    if paths is not None:
        body['paths'] = paths
    if methods is not None:
        body['methods'] = methods
    request.content = json.dumps(body)
    resp = transport.send(request)
    return resp.deserialized_content