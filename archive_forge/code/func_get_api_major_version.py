import abc
import base64
import functools
import hashlib
import json
import threading
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
def get_api_major_version(self, session, service_type=None, interface=None, region_name=None, service_name=None, version=None, allow=None, allow_version_hack=True, skip_discovery=False, discover_versions=False, min_version=None, max_version=None, **kwargs):
    """Return the major API version for a service.

        If a valid token is not present then a new one will be fetched using
        the session and kwargs.

        version, min_version and max_version can all be given either as a
        string or a tuple.

        Valid interface types: `public` or `publicURL`,
                               `internal` or `internalURL`,
                               `admin` or 'adminURL`

        :param session: A session object that can be used for communication.
        :type session: keystoneauth1.session.Session
        :param string service_type: The type of service to lookup the endpoint
                                    for. This plugin will return None (failure)
                                    if service_type is not provided.
        :param interface: Type of endpoint. Can be a single value or a list
                          of values. If it's a list of values, they will be
                          looked for in order of preference. Can also be
                          `keystoneauth1.plugin.AUTH_INTERFACE` to indicate
                          that the auth_url should be used instead of the
                          value in the catalog. (optional, defaults to public)
        :param string region_name: The region the endpoint should exist in.
                                   (optional)
        :param string service_name: The name of the service in the catalog.
                                   (optional)
        :param version: The minimum version number required for this
                        endpoint. (optional)
        :param dict allow: Extra filters to pass when discovering API
                           versions. (optional)
        :param bool allow_version_hack: Allow keystoneauth to hack up catalog
                                        URLS to support older schemes.
                                        (optional, default True)
        :param bool skip_discovery: Whether to skip version discovery even
                                    if a version has been given. This is useful
                                    if endpoint_override or similar has been
                                    given and grabbing additional information
                                    about the endpoint is not useful.
        :param bool discover_versions: Whether to get version metadata from
                                       the version discovery document even
                                       if it's not neccessary to fulfill the
                                       major version request. Defaults to False
                                       because get_endpoint doesn't need
                                       metadata. (optional, defaults to False)
        :param min_version: The minimum version that is acceptable. Mutually
                            exclusive with version. If min_version is given
                            with no max_version it is as if max version is
                            'latest'. (optional)
        :param max_version: The maximum version that is acceptable. Mutually
                            exclusive with version. If min_version is given
                            with no max_version it is as if max version is
                            'latest'. (optional)

        :raises keystoneauth1.exceptions.http.HttpError: An error from an
                                                         invalid HTTP response.

        :return: The major version of the API of the service discovered.
        :rtype: tuple or None

        .. note:: Implementation notes follow. Users should not need to wrap
                  their head around these implementation notes.
                  `get_api_major_version` should do what is expected with the
                  least possible cost while still consistently returning a
                  value if possible.

        There are many cases when major version can be satisfied
        without actually calling the discovery endpoint (like when the version
        is in the url). If the user has a cloud with the versioned endpoint
        ``https://volume.example.com/v3`` in the catalog for the
        ``block-storage`` service and they do::

          client = adapter.Adapter(
              session, service_type='block-storage', min_version=2,
              max_version=3)
          volume_version = client.get_api_major_version()

        The version actually be returned with no api calls other than getting
        the token. For that reason, :meth:`.get_api_major_version` first
        calls :meth:`.get_endpoint_data` with ``discover_versions=False``.

        If their catalog has an unversioned endpoint
        ``https://volume.example.com`` for the ``block-storage`` service
        and they do this::

          client = adapter.Adapter(session, service_type='block-storage')

        client is now set up to "use whatever is in the catalog". Since the
        url doesn't have a version, :meth:`.get_endpoint_data` with
        ``discover_versions=False`` will result in ``api_version=None``.
        (No version was requested so it didn't need to do the round trip)

        In order to find out what version the endpoint actually is, we must
        make a round trip. Therefore, if ``api_version`` is ``None`` after
        the first call, :meth:`.get_api_major_version` will make a second
        call to :meth:`.get_endpoint_data` with ``discover_versions=True``.

        """
    allow = allow or {}
    min_version, max_version = discover._normalize_version_args(version, min_version, max_version, service_type=service_type)
    get_endpoint_data = functools.partial(self.get_endpoint_data, session, service_type=service_type, interface=interface, region_name=region_name, service_name=service_name, allow=allow, min_version=min_version, max_version=max_version, skip_discovery=skip_discovery, allow_version_hack=allow_version_hack, **kwargs)
    data = get_endpoint_data(discover_versions=discover_versions)
    if (not data or not data.api_version) and (not discover_versions):
        data = get_endpoint_data(discover_versions=True)
    if not data:
        return None
    return data.api_version