import contextlib
import hashlib
import os
from oslo_utils import reflection
from oslo_utils import strutils
from troveclient.compat import exceptions
from troveclient.compat import utils
@contextlib.contextmanager
def completion_cache(self, cache_type, obj_class, mode):
    """Bash-completion cache.

        The completion cache store items that can be used for bash
        autocompletion, like UUIDs or human-friendly IDs.

        A resource listing will clear and repopulate the cache.

        A resource create will append to the cache.

        Delete is not handled because listings are assumed to be performed
        often enough to keep the cache reasonably up-to-date.
        """
    base_dir = utils.env('REDDWARFCLIENT_ID_CACHE_DIR', default='~/.troveclient')
    username = utils.env('OS_USERNAME', 'USERNAME')
    url = utils.env('OS_URL', 'SERVICE_URL')
    uniqifier = hashlib.md5(username + url).hexdigest()
    cache_dir = os.path.expanduser(os.path.join(base_dir, uniqifier))
    try:
        os.makedirs(cache_dir, 493)
    except OSError:
        pass
    resource = obj_class.__name__.lower()
    filename = '%s-%s-cache' % (resource, cache_type.replace('_', '-'))
    path = os.path.join(cache_dir, filename)
    cache_attr = '_%s_cache' % cache_type
    try:
        setattr(self, cache_attr, open(path, mode))
    except IOError:
        pass
    try:
        yield
    finally:
        cache = getattr(self, cache_attr, None)
        if cache:
            cache.close()
            delattr(self, cache_attr)