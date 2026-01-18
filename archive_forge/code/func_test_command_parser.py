import argparse
from accelerate.test_utils import execute_subprocess_async, path_in_accelerate_package
def test_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser('test')
    else:
        parser = argparse.ArgumentParser('Accelerate test command')
    parser.add_argument('--config_file', default=None, help="The path to use to store the config file. Will default to a file named default_config.yaml in the cache location, which is the content of the environment `HF_HOME` suffixed with 'accelerate', or if you don't have such an environment variable, your cache directory ('~/.cache' or the content of `XDG_CACHE_HOME`) suffixed with 'huggingface'.")
    if subparsers is not None:
        parser.set_defaults(func=test_command)
    return parser