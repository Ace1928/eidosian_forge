def error_for(response, content):
    """Turn an HTTP response into an HTTPError subclass.

    :return: None if the response code is 1xx, 2xx or 3xx. Otherwise,
    an instance of an appropriate HTTPError subclass (or HTTPError
    if nothing else is appropriate.
    """
    http_errors_by_status_code = {400: BadRequest, 401: Unauthorized, 404: NotFound, 405: MethodNotAllowed, 409: Conflict, 412: PreconditionFailed}
    if response.status // 100 <= 3:
        return None
    else:
        cls = http_errors_by_status_code.get(response.status, HTTPError)
    if cls is HTTPError:
        if response.status // 100 == 5:
            cls = ServerError
        elif response.status // 100 == 4:
            cls = ClientError
    return cls(response, content)