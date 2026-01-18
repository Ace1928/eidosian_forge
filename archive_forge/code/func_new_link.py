import unittest
from traits.api import (
def new_link(self, lt, cur, value):
    link = Link(value=value, next=cur.next, prev=cur)
    cur.next.prev = link
    lt.trait_set(exp_object=cur, exp_name='next', exp_old=cur.next, exp_new=link)
    cur.next = link
    return link