import pytest
from panel.pane import Markdown
from panel.template import FastListTemplate
from panel.tests.util import serve_component
def test_fast_list_template_updates(page):
    tmpl = FastListTemplate()
    md = Markdown('Initial')
    tmpl.main.append(md)
    serve_component(page, tmpl)
    expect(page.locator('.markdown').locator('div')).to_have_text('Initial\n')
    md.object = 'Updated'
    expect(page.locator('.markdown').locator('div')).to_have_text('Updated\n')