import platform
from google.auth import version
def add_metric_header(headers, metric_header_value):
    """Add x-goog-api-client header with the given value.

    Args:
        headers (Mapping[str, str]): The headers to which we will add the
            metric header.
        metric_header_value (Optional[str]): If value is None, do nothing;
            if headers already has a x-goog-api-client header, append the value
            to the existing header; otherwise add a new x-goog-api-client
            header with the given value.
    """
    if not metric_header_value:
        return
    if API_CLIENT_HEADER not in headers:
        headers[API_CLIENT_HEADER] = metric_header_value
    else:
        headers[API_CLIENT_HEADER] += ' ' + metric_header_value