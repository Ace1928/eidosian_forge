import re
from contextlib import contextmanager
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Iterator, Any, Dict, Union, Tuple, Optional, List, cast
import httpx
from qcs_api_client.client import QCSClientConfiguration
from pyquil.api._errors import ApiError, UnknownApiError, TooManyQubitsError, error_mapping
def get_wavefunction(self, request: GetWavefunctionRequest) -> GetWavefunctionResponse:
    """
        Run a program and retrieve the resulting wavefunction.
        """
    payload: Dict[str, Any] = {'type': 'wavefunction', 'compiled-quil': request.program}
    if request.measurement_noise is not None:
        payload['measurement-noise'] = request.measurement_noise
    if request.gate_noise is not None:
        payload['gate-noise'] = request.gate_noise
    if request.seed is not None:
        payload['rng-seed'] = request.seed
    return GetWavefunctionResponse(wavefunction=self._post_json(payload).content)