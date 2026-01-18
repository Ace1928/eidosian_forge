import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def generate_ir(self, nodes, edges):
    m = ir.Module()
    incref_fn = self.make_incref(m)
    decref_fn = self.make_decref(m)
    switcher_fn = self.make_switcher(m)
    brancher_fn = self.make_brancher(m)
    fnty = ir.FunctionType(ir.VoidType(), [ptr_ty])
    fn = ir.Function(m, fnty, name='main')
    [ptr] = fn.args
    ptr.name = 'mem'
    bbmap = {}
    for bb in edges:
        bbmap[bb] = fn.append_basic_block(bb)
    builder = ir.IRBuilder()
    for bb, jump_targets in edges.items():
        builder.position_at_end(bbmap[bb])
        for action in nodes[bb]:
            if action == 'incref':
                builder.call(incref_fn, [ptr])
            elif action == 'decref':
                builder.call(decref_fn, [ptr])
            else:
                raise AssertionError('unreachable')
        n_targets = len(jump_targets)
        if n_targets == 0:
            builder.ret_void()
        elif n_targets == 1:
            [dst] = jump_targets
            builder.branch(bbmap[dst])
        elif n_targets == 2:
            [left, right] = jump_targets
            sel = builder.call(brancher_fn, ())
            builder.cbranch(sel, bbmap[left], bbmap[right])
        elif n_targets > 2:
            sel = builder.call(switcher_fn, ())
            [head, *tail] = jump_targets
            sw = builder.switch(sel, default=bbmap[head])
            for i, dst in enumerate(tail):
                sw.add_case(sel.type(i), bbmap[dst])
        else:
            raise AssertionError('unreachable')
    return m