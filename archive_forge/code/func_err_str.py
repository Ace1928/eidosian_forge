from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def err_str(buf_num_str):
    return f"Object of type '{type(val)}' returned a result object equal to auxiliary_buffer{buf_num_str}. This type violates the contract that appears in apply_channel's documentation."