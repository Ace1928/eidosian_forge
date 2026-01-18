from io import BytesIO
from .lazy_import import lazy_import
from breezy.i18n import gettext
from . import errors, urlutils
from .trace import note
from .transport import (do_catching_redirections, get_transport,
def read_mergeable_from_transport(transport, filename, _do_directive=True):

    def get_bundle(transport):
        return (BytesIO(transport.get_bytes(filename)), transport)

    def redirected_transport(transport, exception, redirection_notice):
        note(redirection_notice)
        url, filename = urlutils.split(exception.target, exclude_trailing_slash=False)
        if not filename:
            raise errors.NotABundle(gettext('A directory cannot be a bundle'))
        return get_transport_from_url(url)
    try:
        bytef, transport = do_catching_redirections(get_bundle, transport, redirected_transport)
    except errors.TooManyRedirections:
        raise errors.NotABundle(transport.clone(filename).base)
    except (errors.ConnectionReset, errors.ConnectionError) as e:
        raise
    except (errors.TransportError, errors.PathError) as e:
        raise errors.NotABundle(str(e))
    except OSError as e:
        raise errors.NotABundle(str(e))
    if _do_directive:
        from .merge_directive import MergeDirective
        try:
            return (MergeDirective.from_lines(bytef), transport)
        except errors.NotAMergeDirective:
            bytef.seek(0)
    from .bzr.bundle import serializer as _serializer
    return (_serializer.read_bundle(bytef), transport)