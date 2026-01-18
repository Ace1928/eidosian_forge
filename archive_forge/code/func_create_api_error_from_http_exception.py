import requests
def create_api_error_from_http_exception(e):
    """
    Create a suitable APIError from requests.exceptions.HTTPError.
    """
    response = e.response
    try:
        explanation = response.json()['message']
    except ValueError:
        explanation = (response.content or '').strip()
    cls = APIError
    if response.status_code == 404:
        explanation_msg = (explanation or '').lower()
        if any((fragment in explanation_msg for fragment in _image_not_found_explanation_fragments)):
            cls = ImageNotFound
        else:
            cls = NotFound
    raise cls(e, response=response, explanation=explanation) from e