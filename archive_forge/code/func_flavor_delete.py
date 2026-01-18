import json
import zaqarclient.transport.errors as errors
def flavor_delete(transport, request, name):
    """Deletes the flavor `name`

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param name: Flavor reference name.
    :type name: str
    """
    request.operation = 'flavor_delete'
    request.params['flavor_name'] = name
    transport.send(request)