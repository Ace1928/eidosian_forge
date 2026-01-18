import errno
import logging
import os
import subprocess
import tempfile
import time
import grpc
import pkg_resources
from tensorboard.data import grpc_provider
from tensorboard.data import ingester
from tensorboard.data.proto import data_provider_pb2
from tensorboard.util import tb_logging
def get_server_binary():
    """Get `ServerBinary` info or raise `NoDataServerError`."""
    env_result = os.environ.get(_ENV_DATA_SERVER_BINARY)
    if env_result:
        logging.info('Server binary (from env): %s', env_result)
        if not os.path.isfile(env_result):
            raise NoDataServerError('Found environment variable %s=%s, but no such file exists.' % (_ENV_DATA_SERVER_BINARY, env_result))
        return ServerBinary(env_result, version=None)
    bundle_result = os.path.join(os.path.dirname(__file__), 'server', 'server')
    if os.path.exists(bundle_result):
        logging.info('Server binary (from bundle): %s', bundle_result)
        return ServerBinary(bundle_result, version=None)
    try:
        import tensorboard_data_server
    except ImportError:
        pass
    else:
        pkg_result = tensorboard_data_server.server_binary()
        version = tensorboard_data_server.__version__
        logging.info('Server binary (from Python package v%s): %s', version, pkg_result)
        if pkg_result is None:
            raise NoDataServerError('TensorBoard data server not supported on this platform.')
        return ServerBinary(pkg_result, version)
    raise NoDataServerError('TensorBoard data server not found. This mode is experimental. If building from source, pass --define=link_data_server=true.')