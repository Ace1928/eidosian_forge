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
def from_protobuf(cls, filepath: str, as_obj: bool=True, **kwargs) -> Union[ModelServerConfig, List[TFSModelConfig]]:
    filepath = get_pathlike(filepath)
    assert filepath.exists()
    serv_config = ModelServerConfig()
    serv_config.FromString(filepath.read_bytes())
    if not as_obj:
        return serv_config
    model_configs = []
    for model in serv_config.model_config_list.config:
        versions = [TFSModelVersion(step=v, label=k) for k, v in model.version_labels.items()]
        model_config = TFSModelConfig(name=model.name, base_path=model.base_path, model_platform=model.model_platform, model_versions=versions)
        model_configs.append(model_config)
    return model_configs