from automaton import exceptions as excp
from automaton import runners
from taskflow.engines.action_engine import builder
from taskflow.engines.action_engine import compiler
from taskflow.engines.action_engine import executor
from taskflow.engines.action_engine import runtime
from taskflow.patterns import linear_flow as lf
from taskflow import states as st
from taskflow import storage
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import notifier
from taskflow.utils import persistence_utils as pu
def test_builder_automatic_process_failure(self):
    flow = lf.Flow('root')
    tasks = test_utils.make_many(1, task_cls=test_utils.NastyFailingTask)
    flow.add(*tasks)
    runtime, machine, memory, machine_runner = self._make_machine(flow, initial_state=st.RUNNING)
    transitions = list(machine_runner.run_iter(builder.START))
    self.assertEqual((builder.GAME_OVER, st.FAILURE), transitions[-1])
    self.assertEqual(1, len(memory.failures))