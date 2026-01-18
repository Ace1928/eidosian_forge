import functools
def proxy_method_directly(*attrs):

    def cls_builder(cls):
        for attr_name in attrs:
            setattr(cls, attr_name, _make_proxy_method(attr_name))
        return cls
    return cls_builder