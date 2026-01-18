import abc
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union
import pkg_resources
import google.auth
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials
from google.oauth2 import service_account # type: ignore
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import empty_pb2
@property
def delete_quantum_reservation(self) -> Callable[[engine.DeleteQuantumReservationRequest], Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]]]:
    raise NotImplementedError()