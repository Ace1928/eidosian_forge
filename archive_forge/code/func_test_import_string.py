from pytest import raises
from graphene import ObjectType, String
from ..module_loading import import_string, lazy_import
def test_import_string():
    MyString = import_string('graphene.String')
    assert MyString == String
    MyObjectTypeMeta = import_string('graphene.ObjectType', '__doc__')
    assert MyObjectTypeMeta == ObjectType.__doc__