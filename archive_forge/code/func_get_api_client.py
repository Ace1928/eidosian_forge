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
def get_api_client(api_endpoint=None):
    server_info = _get_server_info(api_endpoint=api_endpoint)
    _handle_server_info(server_info)
    channel_creds = grpc.ssl_channel_credentials()
    credentials = auth.CredentialsStore().read_credentials()
    if credentials:
        channel_creds = grpc.composite_channel_credentials(channel_creds, auth.id_token_call_credentials(credentials))
    channel = grpc.secure_channel(server_info.api_server.endpoint, channel_creds)
    return export_service_pb2_grpc.TensorBoardExporterServiceStub(channel)