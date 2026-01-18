import pytest
from panel.widgets.icon import ToggleIcon
def test_empty_icon(self):
    with pytest.raises(ValueError, match='The icon parameter must not '):
        ToggleIcon(icon='')