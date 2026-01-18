import traceback
from contextlib import contextmanager
from typing import List, Any, Dict
from ._compatibility import compatibility
@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr):
    global current_meta
    if should_preserve_node_meta:
        current_meta['prev_grad_fn_seq_nr'] = current_meta.get('grad_fn_seq_nr', None)
        current_meta['prev_in_grad_fn'] = current_meta.get('in_grad_fn', None)
        current_meta['grad_fn_seq_nr'] = seq_nr
        current_meta['in_grad_fn'] = True