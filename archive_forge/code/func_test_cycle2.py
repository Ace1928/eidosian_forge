import unittest
from traits.api import (
def test_cycle2(self):
    lt = LinkTest(tc=self, head=self.build_list())
    handlers = [lt.arg_check0, lt.arg_check1, lt.arg_check2, lt.arg_check3, lt.arg_check4]
    nh = len(handlers)
    self.multi_register(lt, handlers, 'head.[next,prev]*.value')
    cur = lt.head
    for i in range(4):
        lt.trait_set(exp_object=cur, exp_name='value', exp_old=10 * i, exp_new=10 * i + 1)
        cur.value = 10 * i + 1
        cur = cur.next
    self.assertEqual(lt.calls, 4 * nh)
    self.multi_register(lt, handlers, 'head.[next,prev]*.value', remove=True)
    cur = lt.head
    for i in range(4):
        cur.value = 10 * i + 2
        cur = cur.next
    self.assertEqual(lt.calls, 4 * nh)