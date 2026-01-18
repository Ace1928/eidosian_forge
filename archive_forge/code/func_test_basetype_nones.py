from ..base import BaseOptions, BaseType
def test_basetype_nones():

    class MyBaseType(CustomType):
        """Documentation"""

        class Meta:
            name = None
            description = None
    assert isinstance(MyBaseType._meta, CustomOptions)
    assert MyBaseType._meta.name == 'MyBaseType'
    assert MyBaseType._meta.description == 'Documentation'