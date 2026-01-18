from ..field import Field
from ..scalars import String
def test_mounted_type_custom():
    unmounted = String(metadata={'hey': 'yo!'})
    mounted = CustomField.mounted(unmounted)
    assert isinstance(mounted, CustomField)
    assert mounted.type == String
    assert mounted.metadata == {'hey': 'yo!'}