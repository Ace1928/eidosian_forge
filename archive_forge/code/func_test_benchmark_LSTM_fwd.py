import timeit
import numpy
import pytest
from thinc.api import LSTM, NumpyOps, Ops, PyTorchLSTM, fix_random_seed, with_padded
from thinc.compat import has_torch
@pytest.mark.skip
def test_benchmark_LSTM_fwd():
    nO = 128
    nI = 128
    n_batch = 1000
    batch_size = 30
    seq_len = 30
    lengths = numpy.random.normal(scale=10, loc=30, size=n_batch * batch_size)
    lengths = numpy.maximum(lengths, 1)
    batches = []
    uniform_lengths = False
    model = with_padded(LSTM(nO, nI)).initialize()
    for batch_lengths in model.ops.minibatch(batch_size, lengths):
        batch_lengths = list(batch_lengths)
        if uniform_lengths:
            seq_len = max(batch_lengths)
            batch = [numpy.asarray(numpy.random.uniform(0.0, 1.0, (int(seq_len), nI)), dtype='f') for _ in batch_lengths]
        else:
            batch = [numpy.asarray(numpy.random.uniform(0.0, 1.0, (int(seq_len), nI)), dtype='f') for seq_len in batch_lengths]
        batches.append(batch)
    start = timeit.default_timer()
    for Xs in batches:
        ys, bp_ys = model.begin_update(list(Xs))
    end = timeit.default_timer()
    n_samples = n_batch * batch_size
    print('--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---' % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))