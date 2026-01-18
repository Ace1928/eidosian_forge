import pytest
from panel.pane import Markdown
from panel.template import FastListTemplate
from panel.tests.util import serve_component
def test_fast_list_template_no_console_errors(page):
    tmpl = FastListTemplate()
    md = Markdown('Initial')
    tmpl.main.append(md)
    msgs, _ = serve_component(page, tmpl)
    expect(page.locator('.markdown').locator('div')).to_have_text('Initial\n')
    known_messages = ["setting log level to: 'info'", 'Websocket connection 0 is now open', 'document idle at', 'items were rendered successfully']
    assert len([msg for msg in msgs if not any((known in msg.text for known in known_messages))]) == 0