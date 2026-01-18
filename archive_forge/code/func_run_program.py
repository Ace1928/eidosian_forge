from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, cast, Tuple, Union, List, Any
from attr import field
import rpcq
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from pyquil.api import EngagementManager
from pyquil._version import DOCS_URL
def run_program(self, request: RunProgramRequest) -> RunProgramResponse:
    """
        Run a program on a QPU.
        """
    rpcq_request = rpcq.messages.QPURequest(id=request.id, program=request.program, patch_values=request.patch_values)
    job_id = self._rpcq_request('execute_qpu_request', request=rpcq_request, priority=request.priority, user=None)
    return RunProgramResponse(job_id=job_id)