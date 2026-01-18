import duet
import pytest
import cirq
def test_flatten_jobs_terminate_from_collector():
    sent = False
    received = []

    class TestCollector(cirq.Collector):

        def next_job(self):
            nonlocal sent
            if sent:
                return
            sent = True
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit(cirq.H(q), cirq.measure(q))
            a = cirq.CircuitSampleJob(circuit=circuit, repetitions=10, tag='test')
            b = cirq.CircuitSampleJob(circuit=circuit, repetitions=10, tag='test')
            return [[a, None], [[[None]]], [[[]]], b]

        def on_job_result(self, job, result):
            received.append(job.tag)
    TestCollector().collect(sampler=cirq.Simulator(), concurrency=5)
    assert received == ['test'] * 2