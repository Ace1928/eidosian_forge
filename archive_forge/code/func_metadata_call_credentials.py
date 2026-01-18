import threading  # pylint: disable=unused-import
import grpc
from grpc import _auth
from grpc.beta import _client_adaptations
from grpc.beta import _metadata
from grpc.beta import _server_adaptations
from grpc.beta import interfaces  # pylint: disable=unused-import
from grpc.framework.common import cardinality  # pylint: disable=unused-import
from grpc.framework.interfaces.face import face  # pylint: disable=unused-import
def metadata_call_credentials(metadata_plugin, name=None):

    def plugin(context, callback):

        def wrapped_callback(beta_metadata, error):
            callback(_metadata.unbeta(beta_metadata), error)
        metadata_plugin(context, wrapped_callback)
    return grpc.metadata_call_credentials(plugin, name=name)