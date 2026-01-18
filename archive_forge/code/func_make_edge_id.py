from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def make_edge_id(qubit1: int, qubit2: int) -> str:
    return '-'.join([str(qubit) for qubit in sorted([qubit1, qubit2])])