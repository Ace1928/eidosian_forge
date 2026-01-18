from typing import AsyncIterable, AsyncIterator, Awaitable, List, Sequence, Union
import asyncio
import concurrent
from unittest import mock
import duet
import pytest
import google.api_core.exceptions as google_exceptions
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine.stream_manager import (
from cirq_google.cloud import quantum
@pytest.mark.parametrize('error_code, current_request_type', [(Code.PROGRAM_DOES_NOT_EXIST, 'create_quantum_program_and_job'), (Code.PROGRAM_DOES_NOT_EXIST, 'get_quantum_result'), (Code.PROGRAM_ALREADY_EXISTS, 'create_quantum_job'), (Code.PROGRAM_ALREADY_EXISTS, 'get_quantum_result'), (Code.JOB_DOES_NOT_EXIST, 'create_quantum_program_and_job'), (Code.JOB_DOES_NOT_EXIST, 'create_quantum_job'), (Code.JOB_ALREADY_EXISTS, 'get_quantum_result')])
def test_get_retry_request_or_raise_expects_stream_error(self, error_code, current_request_type):
    create_quantum_program_and_job_request = quantum.QuantumRunStreamRequest(create_quantum_program_and_job=quantum.CreateQuantumProgramAndJobRequest())
    create_quantum_job_request = quantum.QuantumRunStreamRequest(create_quantum_job=quantum.CreateQuantumJobRequest())
    get_quantum_result_request = quantum.QuantumRunStreamRequest(get_quantum_result=quantum.GetQuantumResultRequest())
    if current_request_type == 'create_quantum_program_and_job':
        current_request = create_quantum_program_and_job_request
    elif current_request_type == 'create_quantum_job':
        current_request = create_quantum_job_request
    elif current_request_type == 'get_quantum_result':
        current_request = get_quantum_result_request
    with pytest.raises(StreamError):
        _get_retry_request_or_raise(quantum.StreamError(code=error_code), current_request, create_quantum_program_and_job_request, create_quantum_job_request, get_quantum_result_request)