from keystoneauth1 import adapter
from keystoneauth1.loading import _utils
from keystoneauth1.loading import base
def register_service_argparse_arguments(*args, **kwargs):
    return adapter.register_service_adapter_argparse_arguments(*args, **kwargs)