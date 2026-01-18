from __future__ import annotations
import inspect
import pytest
import dask
from dask import delayed
from dask.base import collections_to_dsk, key_split, visualize_dsk
from dask.core import get_deps
from dask.order import _connecting_to_roots, diagnostics, ndependencies, order
from dask.utils_test import add, inc
def test_eager_to_compute_dependent_to_free_parent():
    """https://github.com/dask/dask/pull/7929

    This graph begins with many motifs like the following:

    |      |
    c1    c2
      \\ /
       b
       |
       a

    We want to compute c2 and c3 pretty close together, because if we choose to
    compute c1, then we should also compute c2 so we can release b.  Being
    greedy here allows us to release memory sooner and be more globally optimal.
    """
    dsk = {'a00': (f, 'a06', 'a08'), 'a01': (f, 'a28', 'a26'), 'a02': (f, 'a24', 'a21'), 'a03': (f, 'a22', 'a25'), 'a04': (f, 'a29', 'a20'), 'a05': (f, 'a23', 'a27'), 'a06': (f, 'a04', 'a02'), 'a07': (f, 'a00', 'a01'), 'a08': (f, 'a05', 'a03'), 'a09': (f, 'a43'), 'a10': (f, 'a36'), 'a11': (f, 'a33'), 'a12': (f, 'a47'), 'a13': (f, 'a44'), 'a14': (f, 'a42'), 'a15': (f, 'a37'), 'a16': (f, 'a48'), 'a17': (f, 'a49'), 'a18': (f, 'a35'), 'a19': (f, 'a46'), 'a20': (f, 'a55'), 'a21': (f, 'a53'), 'a22': (f, 'a60'), 'a23': (f, 'a54'), 'a24': (f, 'a59'), 'a25': (f, 'a56'), 'a26': (f, 'a61'), 'a27': (f, 'a52'), 'a28': (f, 'a57'), 'a29': (f, 'a58'), 'a30': (f, 'a19'), 'a31': (f, 'a07'), 'a32': (f, 'a30', 'a31'), 'a33': (f, 'a58'), 'a34': (f, 'a11', 'a09'), 'a35': (f, 'a60'), 'a36': (f, 'a52'), 'a37': (f, 'a61'), 'a38': (f, 'a14', 'a10'), 'a39': (f, 'a38', 'a40'), 'a40': (f, 'a18', 'a17'), 'a41': (f, 'a34', 'a50'), 'a42': (f, 'a54'), 'a43': (f, 'a55'), 'a44': (f, 'a53'), 'a45': (f, 'a16', 'a15'), 'a46': (f, 'a51', 'a45'), 'a47': (f, 'a59'), 'a48': (f, 'a57'), 'a49': (f, 'a56'), 'a50': (f, 'a12', 'a13'), 'a51': (f, 'a41', 'a39'), 'a52': (f, 'a62'), 'a53': (f, 'a68'), 'a54': (f, 'a70'), 'a55': (f, 'a67'), 'a56': (f, 'a71'), 'a57': (f, 'a64'), 'a58': (f, 'a65'), 'a59': (f, 'a63'), 'a60': (f, 'a69'), 'a61': (f, 'a66'), 'a62': (f, f), 'a63': (f, f), 'a64': (f, f), 'a65': (f, f), 'a66': (f, f), 'a67': (f, f), 'a68': (f, f), 'a69': (f, f), 'a70': (f, f), 'a71': (f, f)}
    o = order(dsk)
    assert_topological_sort(dsk, o)
    _, pressure = diagnostics(dsk, o=o)
    assert max(pressure) <= 8
    shallow_roots = ['a64', 'a66']
    deep_roots1 = ['a68', 'a63', 'a65', 'a67']
    reducers_1 = ['a06', 'a41']
    deep_roots2 = ['a69', 'a70', 'a71', 'a62']
    reducers_2 = ['a39', 'a00']
    for deep_roots in [deep_roots1, deep_roots2]:
        assert max((o[r] for r in deep_roots)) < min((o[r] for r in shallow_roots))
    for reducer in [reducers_1, reducers_2]:
        for red in reducer:
            assert o[red] < min((o[r] for r in shallow_roots))