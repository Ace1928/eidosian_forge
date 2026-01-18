import sys
from openstack.config.loader import OpenStackConfig  # noqa
def get_cloud_region(service_key=None, options=None, app_name=None, app_version=None, load_yaml_config=True, load_envvars=True, **kwargs):
    config = OpenStackConfig(load_yaml_config=load_yaml_config, load_envvars=load_envvars, app_name=app_name, app_version=app_version)
    if options:
        config.register_argparse_arguments(options, sys.argv, service_key)
        parsed_options = options.parse_known_args(sys.argv)
    else:
        parsed_options = None
    return config.get_one(options=parsed_options, **kwargs)