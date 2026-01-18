import csv
import datetime
import os
from collections import OrderedDict, defaultdict
from typing import IO, Any, Dict, List, Optional, Set
from tabulate import tabulate
import onnx
from onnx import GraphProto, defs, helper
def add_proto(self, proto: onnx.ModelProto, bucket: str, is_model: bool) -> None:
    assert isinstance(proto, onnx.ModelProto)
    self.add_model(proto, bucket, is_model)