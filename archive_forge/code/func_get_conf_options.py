from keystoneauth1 import adapter
from keystoneauth1.loading import _utils
from keystoneauth1.loading import base
@staticmethod
def get_conf_options(include_deprecated=True, deprecated_opts=None):
    """Get oslo_config options that are needed for a :py:class:`.Adapter`.

        These may be useful without being registered for config file generation
        or to manipulate the options before registering them yourself.

        The options that are set are:
            :service_type:      The default service_type for URL discovery.
            :service_name:      The default service_name for URL discovery.
            :interface:         The default interface for URL discovery.
                                (deprecated)
            :valid_interfaces:  List of acceptable interfaces for URL
                                discovery. Can be a list of any of
                                'public', 'internal' or 'admin'.
            :region_name:       The default region_name for URL discovery.
            :endpoint_override: Always use this endpoint URL for requests
                                for this client.
            :version:           The minimum version restricted to a given Major
                                API. Mutually exclusive with min_version and
                                max_version.
            :min_version:       The minimum major version of a given API,
                                intended to be used as the lower bound of a
                                range with max_version. Mutually exclusive with
                                version. If min_version is given with no
                                max_version it is as if max version is
                                'latest'.
            :max_version:       The maximum major version of a given API,
                                intended to be used as the upper bound of a
                                range with min_version. Mutually exclusive with
                                version.

        :param include_deprecated: If True (the default, for backward
                                   compatibility), deprecated options are
                                   included in the result.  If False, they are
                                   excluded.
        :param dict deprecated_opts: Deprecated options that should be included
             in the definition of new options. This should be a dict from the
             name of the new option to a list of oslo.DeprecatedOpts that
             correspond to the new option. (optional)

             For example, to support the ``api_endpoint`` option pointing to
             the new ``endpoint_override`` option name::

                 old_opt = oslo_cfg.DeprecatedOpt('api_endpoint', 'old_group')
                 deprecated_opts={'endpoint_override': [old_opt]}

        :returns: A list of oslo_config options.
        """
    cfg = _utils.get_oslo_config()
    if deprecated_opts is None:
        deprecated_opts = {}
    deprecated_opts = {name.replace('_', '-'): opt for name, opt in deprecated_opts.items()}
    opts = [cfg.StrOpt('service-type', deprecated_opts=deprecated_opts.get('service-type'), help='The default service_type for endpoint URL discovery.'), cfg.StrOpt('service-name', deprecated_opts=deprecated_opts.get('service-name'), help='The default service_name for endpoint URL discovery.'), cfg.ListOpt('valid-interfaces', deprecated_opts=deprecated_opts.get('valid-interfaces'), help='List of interfaces, in order of preference, for endpoint URL.'), cfg.StrOpt('region-name', deprecated_opts=deprecated_opts.get('region-name'), help='The default region_name for endpoint URL discovery.'), cfg.StrOpt('endpoint-override', deprecated_opts=deprecated_opts.get('endpoint-override'), help='Always use this endpoint URL for requests for this client. NOTE: The unversioned endpoint should be specified here; to request a particular API version, use the `version`, `min-version`, and/or `max-version` options.'), cfg.StrOpt('version', deprecated_opts=deprecated_opts.get('version'), help='Minimum Major API version within a given Major API version for endpoint URL discovery. Mutually exclusive with min_version and max_version'), cfg.StrOpt('min-version', deprecated_opts=deprecated_opts.get('min-version'), help='The minimum major version of a given API, intended to be used as the lower bound of a range with max_version. Mutually exclusive with version. If min_version is given with no max_version it is as if max version is "latest".'), cfg.StrOpt('max-version', deprecated_opts=deprecated_opts.get('max-version'), help='The maximum major version of a given API, intended to be used as the upper bound of a range with min_version. Mutually exclusive with version.'), cfg.IntOpt('connect-retries', deprecated_opts=deprecated_opts.get('connect-retries'), help='The maximum number of retries that should be attempted for connection errors.'), cfg.FloatOpt('connect-retry-delay', deprecated_opts=deprecated_opts.get('connect-retry-delay'), help='Delay (in seconds) between two retries for connection errors. If not set, exponential retry starting with 0.5 seconds up to a maximum of 60 seconds is used.'), cfg.IntOpt('status-code-retries', deprecated_opts=deprecated_opts.get('status-code-retries'), help='The maximum number of retries that should be attempted for retriable HTTP status codes.'), cfg.FloatOpt('status-code-retry-delay', deprecated_opts=deprecated_opts.get('status-code-retry-delay'), help='Delay (in seconds) between two retries for retriable status codes. If not set, exponential retry starting with 0.5 seconds up to a maximum of 60 seconds is used.'), cfg.ListOpt('retriable-status-codes', deprecated_opts=deprecated_opts.get('retriable-status-codes'), item_type=cfg.types.Integer(), help='List of retriable HTTP status codes that should be retried. If not set default to  [503]')]
    if include_deprecated:
        opts += [cfg.StrOpt('interface', help='The default interface for endpoint URL discovery.', deprecated_for_removal=True, deprecated_reason='Using valid-interfaces is preferrable because it is capable of accepting a list of possible interfaces.')]
    return opts