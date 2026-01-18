import pytest
from panel.widgets.icon import ToggleIcon
def test_custom_values(self):
    icon = ToggleIcon(icon='thumb-down', active_icon='thumb-up', value=True)
    assert icon.icon == 'thumb-down'
    assert icon.active_icon == 'thumb-up'
    assert icon.value