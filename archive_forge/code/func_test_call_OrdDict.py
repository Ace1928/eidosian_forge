import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
def test_call_OrdDict():
    ad = rlc.OrdDict((('a', rinterface.IntSexpVector([2])), ('b', rinterface.IntSexpVector([1])), (None, rinterface.IntSexpVector([5])), ('c', rinterface.IntSexpVector([0]))))
    mylist = rinterface.baseenv['list'].rcall(tuple(ad.items()), rinterface.globalenv)
    names = [x for x in mylist.do_slot('names')]
    for i in range(4):
        assert ('a', 'b', '', 'c')[i] == names[i]