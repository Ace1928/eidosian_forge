import pytest
from playwright.sync_api import expect
from panel.pane import Textual
from panel.tests.util import serve_component, wait_until
def on_button_pressed(self, event: Button.Pressed) -> None:
    clicks.append(event)