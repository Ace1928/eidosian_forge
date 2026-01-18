from dulwich.client import parse_rsync_url
from .. import urlutils
from .refs import ref_to_branch_name
def git_url_to_bzr_url(location, branch=None, ref=None):
    if branch is not None and ref is not None:
        raise ValueError('only specify one of branch or ref')
    url = urlutils.URL.from_string(location)
    if url.scheme not in KNOWN_GIT_SCHEMES and (not url.scheme.startswith('chroot-')):
        try:
            username, host, path = parse_rsync_url(location)
        except ValueError:
            return location
        else:
            url = urlutils.URL(scheme='git+ssh', quoted_user=urlutils.quote(username) if username else None, quoted_password=None, quoted_host=urlutils.quote(host), port=None, quoted_path=urlutils.quote(path, safe='/~'))
        location = str(url)
    elif url.scheme in SCHEME_REPLACEMENT:
        url.scheme = SCHEME_REPLACEMENT[url.scheme]
        location = str(url)
    if ref == b'HEAD':
        ref = branch = None
    if ref:
        try:
            branch = ref_to_branch_name(ref)
        except ValueError:
            branch = None
        else:
            ref = None
    if ref or branch:
        params = {}
        if ref:
            params['ref'] = urlutils.quote_from_bytes(ref, safe='')
        if branch:
            params['branch'] = urlutils.escape(branch, safe='')
        location = urlutils.join_segment_parameters(location, params)
    return location