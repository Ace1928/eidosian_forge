from ..base import BaseOptions, BaseType
def test_basetype_create_extra():
    MyBaseType = CustomType.create_type('MyBaseType', name='Base', description='Desc')
    assert isinstance(MyBaseType._meta, CustomOptions)
    assert MyBaseType._meta.name == 'Base'
    assert MyBaseType._meta.description == 'Desc'