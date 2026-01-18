import typing as ty
from openstack import exceptions
from openstack import resource
def delete_metadata(self, session, keys):
    request = self._prepare_request()
    headers = {key: '' for key in keys}
    response = session.post(request.url, headers=self._calculate_headers(headers))
    exceptions.raise_from_response(response, error_message='Error deleting metadata keys')
    return self