import functools
def proxy_property_directly(*attrs):

    def cls_builder(cls):
        for attr_name in attrs:
            setattr(cls, attr_name, _make_proxy_property(attr_name))
        return cls
    return cls_builder