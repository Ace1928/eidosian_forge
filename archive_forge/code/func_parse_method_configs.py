import collections
import grpc
from google.api_core import exceptions
from google.api_core import retry
from google.api_core import timeout
def parse_method_configs(interface_config, retry_impl=retry.Retry):
    """Creates default retry and timeout objects for each method in a gapic
    interface config.

    DEPRECATED: instantiate retry and timeout classes directly instead.

    Args:
        interface_config (Mapping): The interface config section of the full
            gapic library config. For example, If the full configuration has
            an interface named ``google.example.v1.ExampleService`` you would
            pass in just that interface's configuration, for example
            ``gapic_config['interfaces']['google.example.v1.ExampleService']``.
        retry_impl (Callable): The constructor that creates a retry decorator
            that will be applied to the method based on method configs.

    Returns:
        Mapping[str, MethodConfig]: A mapping of RPC method names to their
            configuration.
    """
    retry_codes_map = {name: retry_codes for name, retry_codes in interface_config.get('retry_codes', {}).items()}
    retry_params_map = {name: retry_params for name, retry_params in interface_config.get('retry_params', {}).items()}
    method_configs = {}
    for method_name, method_params in interface_config.get('methods', {}).items():
        retry_params_name = method_params.get('retry_params_name')
        if retry_params_name is not None:
            retry_params = retry_params_map[retry_params_name]
            retry_ = _retry_from_retry_config(retry_params, retry_codes_map[method_params['retry_codes_name']], retry_impl)
            timeout_ = _timeout_from_retry_config(retry_params)
        else:
            retry_ = None
            timeout_ = timeout.ConstantTimeout(method_params['timeout_millis'] / _MILLIS_PER_SECOND)
        method_configs[method_name] = MethodConfig(retry=retry_, timeout=timeout_)
    return method_configs