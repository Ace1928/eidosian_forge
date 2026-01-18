from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def compiler_isa_to_target_quantum_processor(compiler_isa: CompilerISA) -> TargetQuantumProcessor:
    return TargetQuantumProcessor(isa=compiler_isa.dict(by_alias=True), specs={})