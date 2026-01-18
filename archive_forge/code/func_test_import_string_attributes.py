from pytest import raises
from graphene import ObjectType, String
from ..module_loading import import_string, lazy_import
def test_import_string_attributes():
    with raises(Exception) as exc_info:
        import_string('graphene.String', 'length')
    assert str(exc_info.value) == 'Module "graphene" does not define a "length" attribute inside attribute/class "String"'
    with raises(Exception) as exc_info:
        import_string('graphene.ObjectType', '__class__.length')
    assert str(exc_info.value) == 'Module "graphene" does not define a "__class__.length" attribute inside attribute/class "ObjectType"'
    with raises(Exception) as exc_info:
        import_string('graphene.ObjectType', '__classa__.__base__')
    assert str(exc_info.value) == 'Module "graphene" does not define a "__classa__" attribute inside attribute/class "ObjectType"'