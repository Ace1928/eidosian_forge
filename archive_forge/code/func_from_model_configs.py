from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.text_format import Merge
from ._base import TFSModelConfig, TFSModelVersion, List,  Union, Optional, Any, Dict
from ._base import dataclass, lazyclass
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
@classmethod
def from_model_configs(cls, model_configs: List[TFSModelConfig]) -> ModelServerConfig:
    serv_config = ModelServerConfig()
    model_config_list = serv_config.model_config_list
    for config in model_configs:
        model = model_config_list.config.add()
        model.name = config.name
        model.base_path = config.base_path
        model.model_platform = config.model_platform
        if config.model_versions:
            labels = model.version_labels
            policy = model.model_version_policy
            versions = policy.specific.versions
            for model_version in config.model_versions:
                versions.append(model_version.step)
                labels[model_version.label] = model_version.step
    return serv_config