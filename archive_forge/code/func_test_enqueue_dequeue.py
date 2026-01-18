import pytest
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
import cirq
def test_enqueue_dequeue():
    q = BucketPriorityQueue()
    q.enqueue(5, 'a')
    assert q == BucketPriorityQueue([(5, 'a')])
    q.enqueue(4, 'b')
    assert q == BucketPriorityQueue([(4, 'b'), (5, 'a')])
    assert q.dequeue() == (4, 'b')
    assert q == BucketPriorityQueue([(5, 'a')])
    assert q.dequeue() == (5, 'a')
    assert q == BucketPriorityQueue()
    with pytest.raises(ValueError, match='empty'):
        _ = q.dequeue()