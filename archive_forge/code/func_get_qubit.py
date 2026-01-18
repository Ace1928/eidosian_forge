from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def get_qubit(quantum_processor: CompilerISA, node_id: int) -> Optional[Qubit]:
    return quantum_processor.qubits.get(str(node_id))