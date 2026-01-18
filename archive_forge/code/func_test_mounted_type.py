from ..field import Field
from ..scalars import String
def test_mounted_type():
    unmounted = String()
    mounted = Field.mounted(unmounted)
    assert isinstance(mounted, Field)
    assert mounted.type == String