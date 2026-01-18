import typing as ty
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def call_vendor_passthru(self, session, verb: str, method: str, body: ty.Optional[dict]=None):
    """Call a vendor specific passthru method

        Contents of body are params passed to the hardware driver
        function. Validation happens there. Missing parameters, or
        excess parameters will cause the request to be rejected

        :param session: The session to use for making this request.
        :param method: Vendor passthru method name.
        :param verb: One of GET, POST, PUT, DELETE,
            depending on the driver and method.
        :param body: passed to the vendor function as json body.
        :raises: :exc:`ValueError` if :data:`verb` is not one of
            GET, POST, PUT, DELETE
        :returns: response of method call.
        """
    if verb.upper() not in ['GET', 'PUT', 'POST', 'DELETE']:
        raise ValueError('Invalid verb: {}'.format(verb))
    session = self._get_session(session)
    request = self._prepare_request()
    request.url = utils.urljoin(request.url, f'vendor_passthru?method={method}')
    call = getattr(session, verb.lower())
    response = call(request.url, json=body, headers=request.headers, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed call to method {method} on driver {driver_name}'.format(method=method, driver_name=self.name)
    exceptions.raise_from_response(response, error_message=msg)
    return response