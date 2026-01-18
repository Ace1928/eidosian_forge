import json
import zaqarclient.transport.errors as errors
def message_delete(transport, request, queue_name, message_id, claim_id=None, callback=None):
    """Deletes messages from `queue_name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param queue_name: Queue reference name.
    :type queue_name: str
    :param message_id: Message reference.
    :param message_id: str
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = 'message_delete'
    request.params['queue_name'] = queue_name
    request.params['message_id'] = message_id
    if claim_id:
        request.params['claim_id'] = claim_id
    transport.send(request)