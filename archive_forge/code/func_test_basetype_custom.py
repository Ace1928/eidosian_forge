from ..base import BaseOptions, BaseType
def test_basetype_custom():

    class MyBaseType(CustomType):
        """Documentation"""

        class Meta:
            name = 'Base'
            description = 'Desc'
    assert isinstance(MyBaseType._meta, CustomOptions)
    assert MyBaseType._meta.name == 'Base'
    assert MyBaseType._meta.description == 'Desc'