import json
import zaqarclient.transport.errors as errors
def flavor_get(transport, request, flavor_name, callback=None):
    """Gets flavor data

    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param flavor_name: Flavor reference name.
    :type flavor_name: str

    """
    request.operation = 'flavor_get'
    request.params['flavor_name'] = flavor_name
    resp = transport.send(request)
    return resp.deserialized_content