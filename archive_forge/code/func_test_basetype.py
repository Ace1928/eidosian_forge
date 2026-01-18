from ..base import BaseOptions, BaseType
def test_basetype():

    class MyBaseType(CustomType):
        pass
    assert isinstance(MyBaseType._meta, CustomOptions)
    assert MyBaseType._meta.name == 'MyBaseType'
    assert MyBaseType._meta.description is None