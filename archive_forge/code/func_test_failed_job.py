import duet
import pytest
import cirq
def test_failed_job():

    class FailingSampler:

        async def run_async(self, circuit, repetitions):
            await duet.completed_future(None)
            raise Exception('job failed!')

    class TestCollector(cirq.Collector):

        def next_job(self):
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            return cirq.CircuitSampleJob(circuit=circuit, repetitions=10, tag='test')

        def on_job_result(self, job, result):
            pass
    with pytest.raises(Exception, match='job failed!'):
        TestCollector().collect(sampler=FailingSampler(), max_total_samples=100, concurrency=5)