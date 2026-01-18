import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
def test_arraysequence_operators(self):
    flags = np.seterr(divide='ignore', invalid='ignore')
    SCALARS = [42, 0.5, True, -3, 0]
    CMP_OPS = ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']
    seq = SEQ_DATA['seq'].copy()
    seq_int = SEQ_DATA['seq'].copy()
    seq_int._data = seq_int._data.astype(int)
    seq_bool = SEQ_DATA['seq'].copy() > 30
    ARRSEQS = [seq, seq_int, seq_bool]
    VIEWS = [seq[::2], seq_int[::2], seq_bool[::2]]

    def _test_unary(op, arrseq):
        orig = arrseq.copy()
        seq = getattr(orig, op)()
        assert seq is not orig
        check_arr_seq(seq, [getattr(d, op)() for d in orig])

    def _test_binary(op, arrseq, scalars, seqs, inplace=False):
        for scalar in scalars:
            orig = arrseq.copy()
            seq = getattr(orig, op)(scalar)
            assert (seq is orig) == inplace
            check_arr_seq(seq, [getattr(e, op)(scalar) for e in arrseq])
        for other in seqs:
            orig = arrseq.copy()
            seq = getattr(orig, op)(other)
            assert seq is not SEQ_DATA['seq']
            check_arr_seq(seq, [getattr(e1, op)(e2) for e1, e2 in zip(arrseq, other)])
        orig = arrseq.copy()
        with pytest.raises(ValueError):
            getattr(orig, op)(orig[::2])
        seq1 = ArraySequence(np.arange(10).reshape(5, 2))
        seq2 = ArraySequence(np.arange(15).reshape(5, 3))
        with pytest.raises(ValueError):
            getattr(seq1, op)(seq2)
        seq1 = ArraySequence(np.arange(12).reshape(2, 2, 3))
        seq2 = ArraySequence(np.arange(8).reshape(2, 2, 2))
        with pytest.raises(ValueError):
            getattr(seq1, op)(seq2)
    for op in ['__add__', '__sub__', '__mul__', '__mod__', '__floordiv__', '__truediv__'] + CMP_OPS:
        _test_binary(op, seq, SCALARS, ARRSEQS)
        _test_binary(op, seq_int, SCALARS, ARRSEQS)
        _test_binary(op, seq[::2], SCALARS, VIEWS)
        _test_binary(op, seq_int[::2], SCALARS, VIEWS)
        if op in CMP_OPS:
            continue
        op = f'__i{op.strip('_')}__'
        _test_binary(op, seq, SCALARS, ARRSEQS, inplace=True)
        if op == '__itruediv__':
            continue
        _test_binary(op, seq_int, [42, -3, True, 0], [seq_int, seq_bool, -seq_int], inplace=True)
        with pytest.raises(TypeError):
            _test_binary(op, seq_int, [0.5], [], inplace=True)
        with pytest.raises(TypeError):
            _test_binary(op, seq_int, [], [seq], inplace=True)
    _test_binary('__pow__', seq, [42, -3, True, 0], [seq_int, seq_bool, -seq_int])
    _test_binary('__ipow__', seq, [42, -3, True, 0], [seq_int, seq_bool, -seq_int], inplace=True)
    with pytest.raises(ValueError):
        _test_binary('__pow__', seq_int, [-3], [])
    with pytest.raises(ValueError):
        _test_binary('__ipow__', seq_int, [-3], [], inplace=True)
    for scalar in SCALARS + ARRSEQS:
        seq_int_cp = seq_int.copy()
        with pytest.raises(TypeError):
            seq_int_cp /= scalar
    for op in ('__lshift__', '__rshift__', '__or__', '__and__', '__xor__'):
        _test_binary(op, seq_bool, [42, -3, True, 0], [seq_int, seq_bool, -seq_int])
        with pytest.raises(TypeError):
            _test_binary(op, seq_bool, [0.5], [])
        with pytest.raises(TypeError):
            _test_binary(op, seq, [], [seq])
    for op in ['__neg__', '__abs__']:
        _test_unary(op, seq)
        _test_unary(op, -seq)
        _test_unary(op, seq_int)
        _test_unary(op, -seq_int)
    _test_unary('__abs__', seq_bool)
    _test_unary('__invert__', seq_bool)
    with pytest.raises(TypeError):
        _test_unary('__invert__', seq)
    np.seterr(**flags)