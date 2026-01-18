from pytest import raises
from graphene import ObjectType, String
from ..module_loading import import_string, lazy_import
def test_import_string_class():
    with raises(Exception) as exc_info:
        import_string('graphene.Stringa')
    assert str(exc_info.value) == 'Module "graphene" does not define a "Stringa" attribute/class'