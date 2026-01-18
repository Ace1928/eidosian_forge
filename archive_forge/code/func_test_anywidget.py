import pytest
import traitlets
from bokeh.core.has_props import _default_resolver
from bokeh.model import Model
from panel.layout import Row
from panel.pane.ipywidget import Reacton
from panel.tests.util import serve_component, wait_until
@requires_anywidget()
def test_anywidget(page):

    class CounterWidget(anywidget.AnyWidget):
        _esm = '\n        export function render(view) {\n          let getCount = () => view.model.get("count");\n          let button = document.createElement("button");\n          button.innerHTML = `count is ${getCount()}`;\n          button.addEventListener("click", () => {\n            view.model.set("count", getCount() + 1);\n            view.model.save_changes();\n          });\n          view.model.on("change:count", () => {\n            button.innerHTML = `count is ${getCount()}`;\n          });\n          view.el.appendChild(button);\n        }\n        '
        count = traitlets.Int(0).tag(sync=True)
    counter = CounterWidget()
    serve_component(page, counter)
    page.locator('.lm-Widget button').click()
    wait_until(lambda: counter.count == 1, page)
    page.locator('.lm-Widget button').click()
    wait_until(lambda: counter.count == 2, page)