import warnings
from . import exceptions as exc
from . import misc
from . import normalizers
from . import validators
def authority_info(self):
    """Return a dictionary with the ``userinfo``, ``host``, and ``port``.

        If the authority is not valid, it will raise a
        :class:`~rfc3986.exceptions.InvalidAuthority` Exception.

        :returns:
            ``{'userinfo': 'username:password', 'host': 'www.example.com',
            'port': '80'}``
        :rtype: dict
        :raises rfc3986.exceptions.InvalidAuthority:
            If the authority is not ``None`` and can not be parsed.
        """
    if not self.authority:
        return {'userinfo': None, 'host': None, 'port': None}
    match = self._match_subauthority()
    if match is None:
        raise exc.InvalidAuthority(self.authority.encode(self.encoding))
    matches = match.groupdict()
    host = matches.get('host')
    if host and misc.IPv4_MATCHER.match(host) and (not validators.valid_ipv4_host_address(host)):
        raise exc.InvalidAuthority(self.authority.encode(self.encoding))
    return matches