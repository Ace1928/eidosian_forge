from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_from_custom_template_table(tmpdir):
    p = tmpdir.mkdir('tpl').join('myhtml_table.tpl')
    p.write(dedent('            {% extends "html_table.tpl" %}\n            {% block table %}\n            <h1>{{custom_title}}</h1>\n            {{ super() }}\n            {% endblock table %}'))
    result = Styler.from_custom_template(str(tmpdir.join('tpl')), 'myhtml_table.tpl')
    assert issubclass(result, Styler)
    assert result.env is not Styler.env
    assert result.template_html_table is not Styler.template_html_table
    styler = result(DataFrame({'A': [1, 2]}))
    assert '<h1>My Title</h1>\n\n\n<table' in styler.to_html(custom_title='My Title')