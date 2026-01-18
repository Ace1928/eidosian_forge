from ._utils import AttributeDict
from . import exceptions
def app_strip_relative_path(requests_pathname, path):
    if path is None:
        return None
    if requests_pathname != '/' and (not path.startswith(requests_pathname.rstrip('/'))) or (requests_pathname == '/' and (not path.startswith('/'))):
        raise exceptions.UnsupportedRelativePath(f"\n            Paths that aren't prefixed with requests_pathname_prefix are not supported.\n            You supplied: {path} and requests_pathname_prefix was {requests_pathname}\n            ")
    if requests_pathname != '/' and path.startswith(requests_pathname.rstrip('/')):
        path = path.replace(requests_pathname.rstrip('/'), '', 1)
    return path.strip('/')