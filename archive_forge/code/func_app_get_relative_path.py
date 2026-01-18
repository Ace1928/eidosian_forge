from ._utils import AttributeDict
from . import exceptions
def app_get_relative_path(requests_pathname, path):
    if requests_pathname == '/' and path == '':
        return '/'
    if requests_pathname != '/' and path == '':
        return requests_pathname
    if not path.startswith('/'):
        raise exceptions.UnsupportedRelativePath(f"\n            Paths that aren't prefixed with a leading / are not supported.\n            You supplied: {path}\n            ")
    return '/'.join([requests_pathname.rstrip('/'), path.lstrip('/')])