import functools
def cond_delegate_to_executor(*attrs):

    def cls_builder(cls):
        for attr_name in attrs:
            setattr(cls, attr_name, _make_cond_delegate_method(attr_name))
        return cls
    return cls_builder