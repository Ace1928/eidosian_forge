import logging
from oslo_utils import timeutils
from suds import sudsobject
def get_http_service_request_spec(client_factory, method, uri):
    """Build a HTTP service request spec.

    :param client_factory: factory to get API input specs
    :param method: HTTP method (GET, POST, PUT)
    :param uri: target URL
    """
    http_service_request_spec = client_factory.create('ns0:SessionManagerHttpServiceRequestSpec')
    http_service_request_spec.method = method
    http_service_request_spec.url = uri
    return http_service_request_spec