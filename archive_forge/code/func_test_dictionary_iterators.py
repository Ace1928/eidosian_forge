import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_dictionary_iterators(monkeypatch):

    def stock_method_name(iterwhat):
        """Given a method suffix like "lists" or "values", return the name
        of the dict method that delivers those on the version of Python
        we're running in."""
        if six.PY3:
            return iterwhat
        return 'iter' + iterwhat

    class MyDict(dict):
        if not six.PY3:

            def lists(self, **kw):
                return [1, 2, 3]

        def iterlists(self, **kw):
            return iter([1, 2, 3])
    f = MyDict.iterlists
    del MyDict.iterlists
    setattr(MyDict, stock_method_name('lists'), f)
    d = MyDict(zip(range(10), reversed(range(10))))
    for name in ('keys', 'values', 'items', 'lists'):
        meth = getattr(six, 'iter' + name)
        it = meth(d)
        assert not isinstance(it, list)
        assert list(it) == list(getattr(d, name)())
        pytest.raises(StopIteration, six.advance_iterator, it)
        record = []

        def with_kw(*args, **kw):
            record.append(kw['kw'])
            return old(*args)
        old = getattr(MyDict, stock_method_name(name))
        monkeypatch.setattr(MyDict, stock_method_name(name), with_kw)
        meth(d, kw=42)
        assert record == [42]
        monkeypatch.undo()