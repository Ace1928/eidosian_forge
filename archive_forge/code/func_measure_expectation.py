import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
def measure_expectation(self, request: MeasureExpectationRequest) -> MeasureExpectationResponse:
    """
        Measure expectation value of Pauli operators given a defined state.
        """
    payload: Dict[str, Any] = {'type': 'expectation', 'state-preparation': request.prep_program, 'operators': request.pauli_operators}
    if request.seed is not None:
        payload['rng-seed'] = request.seed
    return MeasureExpectationResponse(expectations=cast(List[float], self._post_json(payload).json()))