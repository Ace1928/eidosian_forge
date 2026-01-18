from __future__ import annotations
from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
import pandas as pd
from pandas import (
import pandas.core.common as com
from pandas.core.frame import (
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats.format import save_to_buffer
from pandas.io.formats.style_render import (
@classmethod
def from_custom_template(cls, searchpath: Sequence[str], html_table: str | None=None, html_style: str | None=None) -> type[Styler]:
    """
        Factory function for creating a subclass of ``Styler``.

        Uses custom templates and Jinja environment.

        .. versionchanged:: 1.3.0

        Parameters
        ----------
        searchpath : str or list
            Path or paths of directories containing the templates.
        html_table : str
            Name of your custom template to replace the html_table template.

            .. versionadded:: 1.3.0

        html_style : str
            Name of your custom template to replace the html_style template.

            .. versionadded:: 1.3.0

        Returns
        -------
        MyStyler : subclass of Styler
            Has the correct ``env``,``template_html``, ``template_html_table`` and
            ``template_html_style`` class attributes set.

        Examples
        --------
        >>> from pandas.io.formats.style import Styler
        >>> EasyStyler = Styler.from_custom_template("path/to/template",
        ...                                          "template.tpl",
        ...                                          )  # doctest: +SKIP
        >>> df = pd.DataFrame({"A": [1, 2]})
        >>> EasyStyler(df)  # doctest: +SKIP

        Please see:
        `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
        """
    loader = jinja2.ChoiceLoader([jinja2.FileSystemLoader(searchpath), cls.loader])

    class MyStyler(cls):
        env = jinja2.Environment(loader=loader)
        if html_table:
            template_html_table = env.get_template(html_table)
        if html_style:
            template_html_style = env.get_template(html_style)
    return MyStyler