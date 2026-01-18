import json
import re
import typing as ty
from requests import exceptions as _rex
def raise_from_response(response, error_message=None):
    """Raise an instance of an HTTPException based on keystoneauth response."""
    if response.status_code < 400:
        return
    cls: ty.Type[SDKException]
    if response.status_code == 400:
        cls = BadRequestException
    elif response.status_code == 403:
        cls = ForbiddenException
    elif response.status_code == 404:
        cls = NotFoundException
    elif response.status_code == 409:
        cls = ConflictException
    elif response.status_code == 412:
        cls = PreconditionFailedException
    else:
        cls = HttpException
    details = None
    content_type = response.headers.get('content-type', '')
    if response.content and 'application/json' in content_type:
        try:
            content = response.json()
            messages = [_extract_message(obj) for obj in content.values()]
            if not any(messages):
                messages = [_extract_message(content)]
            details = '\n'.join((msg for msg in messages if msg))
        except Exception:
            details = response.text
    elif response.content and 'text/html' in content_type:
        messages = []
        for line in response.text.splitlines():
            message = re.sub('<.+?>', '', line.strip())
            if message not in messages:
                messages.append(message)
        details = ': '.join(messages)
    if not details:
        details = response.reason if response.reason else response.text
    http_status = response.status_code
    request_id = response.headers.get('x-openstack-request-id')
    raise cls(message=error_message, response=response, details=details, http_status=http_status, request_id=request_id)