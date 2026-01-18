import json
import zaqarclient.transport.errors as errors
def message_pop(transport, request, queue_name, count, callback=None):
    """Pops out `count` messages from `queue_name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param count: Number of messages to pop.
    :type count: int
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = 'message_delete_many'
    request.params['queue_name'] = queue_name
    request.params['pop'] = count
    resp = transport.send(request)
    return resp.deserialized_content