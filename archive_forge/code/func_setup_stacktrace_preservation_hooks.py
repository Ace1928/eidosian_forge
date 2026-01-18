import collections
from contextlib import contextmanager
from typing import List, Tuple
import torch
import torch.fx.traceback as fx_traceback
def setup_stacktrace_preservation_hooks(roots: List):

    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)
        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)
            yield node

    def get_callback(saved_stack_):

        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False
        return callback

    def get_prehook(stack_, seq_nr):

        def prehook(grad_output):
            global callback_set
            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(get_callback(fx_traceback.format_stack()))
                callback_set = True
            fx_traceback.set_stack_trace(stack_)
            fx_traceback.set_grad_fn_seq_nr(seq_nr)
        return prehook

    def get_posthook(special_stack_, seq_nr):

        def posthook(grad_input, grad_output):
            fx_traceback.set_stack_trace(special_stack_)
            fx_traceback.reset_grad_fn_seq_nr()
        return posthook
    for node in iter_graph(roots):
        forward_node_stack = node.metadata.get('traceback_', [])
        node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))
        special_stack = forward_node_stack.copy()
        special_stack.append('Gradient addition node due to multiple use of tensor around:')
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))