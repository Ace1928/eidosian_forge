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
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_programs_async')
def test_list_programs(list_programs_async):
    prog1 = quantum.QuantumProgram(name='projects/proj/programs/prog-YBGR48THF3JHERZW200804')
    prog2 = quantum.QuantumProgram(name='projects/otherproj/programs/prog-V3ZRTV6TTAFNTYJV200804')
    list_programs_async.return_value = [prog1, prog2]
    result = cg.Engine(project_id='proj').list_programs()
    list_programs_async.assert_called_once_with('proj', created_after=None, created_before=None, has_labels=None)
    assert [(p.program_id, p.project_id, p._program) for p in result] == [('prog-YBGR48THF3JHERZW200804', 'proj', prog1), ('prog-V3ZRTV6TTAFNTYJV200804', 'otherproj', prog2)]