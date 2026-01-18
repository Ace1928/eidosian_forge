import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
def bucket_has_external_output(bucket: Bucket) -> bool:
    nodes_in_bucket = set()
    for node in bucket.nodes:
        nodes_in_bucket.add(node)
        for user in node.users:
            if user not in nodes_in_bucket:
                return True
    return False