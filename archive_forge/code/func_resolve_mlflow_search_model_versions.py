import graphene
import mlflow
from mlflow.server.graphql.graphql_custom_scalars import LongString
from mlflow.utils.proto_json_utils import parse_dict
def resolve_mlflow_search_model_versions(self, info, input):
    input_dict = vars(input)
    request_message = mlflow.protos.model_registry_pb2.SearchModelVersions()
    parse_dict(input_dict, request_message)
    return mlflow.server.handlers.search_model_versions_impl(request_message)