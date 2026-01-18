import contextlib
import taskflow.engines
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import task
from taskflow import test
from taskflow.utils import persistence_utils as p_utils
def test_dual_storage_progress(self):
    fired_events = []

    def notify_me(event_type, details):
        fired_events.append(details.pop('progress'))
    with contextlib.closing(impl_memory.MemoryBackend({})) as be:
        t = ProgressTask('test', 5)
        t.notifier.register(task.EVENT_UPDATE_PROGRESS, notify_me)
        flo = lf.Flow('test')
        flo.add(t)
        b, fd = p_utils.temporary_flow_detail(be)
        e = self._make_engine(flo, flow_detail=fd, backend=be)
        e.run()
        end_progress = e.storage.get_task_progress('test')
        self.assertEqual(1.0, end_progress)
        task_uuid = e.storage.get_atom_uuid('test')
        td = fd.find(task_uuid)
        self.assertEqual(1.0, td.meta['progress'])
        self.assertFalse(td.meta['progress_details'])
        self.assertEqual(6, len(fired_events))