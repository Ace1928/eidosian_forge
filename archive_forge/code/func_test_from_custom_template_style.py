from textwrap import (
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_from_custom_template_style(tmpdir):
    p = tmpdir.mkdir('tpl').join('myhtml_style.tpl')
    p.write(dedent('            {% extends "html_style.tpl" %}\n            {% block style %}\n            <link rel="stylesheet" href="mystyle.css">\n            {{ super() }}\n            {% endblock style %}'))
    result = Styler.from_custom_template(str(tmpdir.join('tpl')), html_style='myhtml_style.tpl')
    assert issubclass(result, Styler)
    assert result.env is not Styler.env
    assert result.template_html_style is not Styler.template_html_style
    styler = result(DataFrame({'A': [1, 2]}))
    assert '<link rel="stylesheet" href="mystyle.css">\n\n<style' in styler.to_html()