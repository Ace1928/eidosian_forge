import datetime
from unittest import mock
import time
import numpy as np
import pytest
import duet
from google.protobuf import any_pb2, timestamp_pb2
from google.protobuf.text_format import Merge
import cirq
import cirq_google
import cirq_google as cg
from cirq_google.api import v1, v2
from cirq_google.engine import util
from cirq_google.cloud import quantum
from cirq_google.engine.engine import EngineContext
def test_bad_program_proto():
    engine = cg.Engine(project_id='project-id', proto_version=cg.engine.engine.ProtoVersion.UNDEFINED)
    with pytest.raises(ValueError, match='invalid (program|run context) proto version'):
        engine.run_sweep(program=_CIRCUIT)
    with pytest.raises(ValueError, match='invalid program proto version'):
        engine.create_program(_CIRCUIT)