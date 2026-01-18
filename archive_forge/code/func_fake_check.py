import torch._subclasses
def fake_check(op, args, kwargs):
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        op(*args, **kwargs)