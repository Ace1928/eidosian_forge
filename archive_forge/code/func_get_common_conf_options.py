from keystoneauth1.loading import base
from keystoneauth1.loading import opts
def get_common_conf_options():
    """Get the oslo_config options common for all auth plugins.

    These may be useful without being registered for config file generation
    or to manipulate the options before registering them yourself.

    The options that are set are:
        :auth_type: The name of the plugin to load.
        :auth_section: The config file section to load options from.

    :returns: A list of oslo_config options.
    """
    return [_AUTH_TYPE_OPT._to_oslo_opt(), _AUTH_SECTION_OPT._to_oslo_opt()]