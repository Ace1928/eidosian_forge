import functools
def cls_builder(cls):
    for attr_name in attrs:
        setattr(cls, attr_name, _make_cond_delegate_method(attr_name))
    return cls