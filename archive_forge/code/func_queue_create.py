import json
import zaqarclient.transport.errors as errors
def queue_create(transport, request, name, metadata=None, callback=None):
    """Creates a queue

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param name: Queue reference name.
    :type name: str
    :param metadata: Queue's metadata object. (>=v1.1)
    :type metadata: `dict`
    :param callback: Optional callable to use as callback.
        If specified, this request will be sent asynchronously.
        (IGNORED UNTIL ASYNC SUPPORT IS COMPLETE)
    :type callback: Callable object.
    """
    request.operation = 'queue_create'
    request.params['queue_name'] = name
    request.content = metadata and json.dumps(metadata)
    resp = transport.send(request)
    return resp.deserialized_content