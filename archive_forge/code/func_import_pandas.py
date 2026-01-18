import sys
import time
import grpc
from tensorboard.data.experimental import base_experiment
from tensorboard.data.experimental import utils as experimental_utils
from tensorboard.uploader import auth
from tensorboard.uploader import util
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.uploader.proto import server_info_pb2
from tensorboard.util import grpc_util
def import_pandas():
    """Import pandas, guarded by a user-friendly error message on failure."""
    try:
        import pandas
    except ImportError:
        raise ImportError('The get_scalars() feature requires the pandas package, which does not seem to be available in your Python environment. You can install it with command:\n\n  pip install pandas\n')
    return pandas