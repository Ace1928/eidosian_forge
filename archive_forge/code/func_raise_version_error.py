from .. import auth, errors, utils
from ..types import ServiceMode
def raise_version_error(param, min_version):
    raise errors.InvalidVersion('{} is not supported in API version < {}'.format(param, min_version))