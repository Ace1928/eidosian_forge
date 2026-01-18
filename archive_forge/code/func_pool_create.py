import json
import zaqarclient.transport.errors as errors
def pool_create(transport, request, pool_name, pool_data):
    """Creates a pool called `pool_name`


    :param transport: Transport instance to use
    :type transport: `transport.base.Transport`
    :param request: Request instance ready to be sent.
    :type request: `transport.request.Request`
    :param pool_name: Pool reference name.
    :type pool_name: str
    :param pool_data: Pool's properties, i.e: weight, uri, options.
    :type pool_data: `dict`
    """
    request.operation = 'pool_create'
    request.params['pool_name'] = pool_name
    request.content = json.dumps(pool_data)
    transport.send(request)