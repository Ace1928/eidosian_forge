import json
from typing import TYPE_CHECKING, Any, Dict, Union
from wandb.proto import wandb_internal_pb2 as pb
def dict_from_proto_list(obj_list: 'RepeatedCompositeFieldContainer') -> Dict[str, Any]:
    return {item.key: json.loads(item.value_json) for item in obj_list}