import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
def estimator_html_repr(estimator):
    """Build a HTML representation of an estimator.

    Read more in the :ref:`User Guide <visualizing_composite_estimators>`.

    Parameters
    ----------
    estimator : estimator object
        The estimator to visualize.

    Returns
    -------
    html: str
        HTML representation of estimator.

    Examples
    --------
    >>> from sklearn.utils._estimator_html_repr import estimator_html_repr
    >>> from sklearn.linear_model import LogisticRegression
    >>> estimator_html_repr(LogisticRegression())
    '<style>...</div>'
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    if not hasattr(estimator, 'fit'):
        status_label = '<span>Not fitted</span>'
        is_fitted_css_class = ''
    else:
        try:
            check_is_fitted(estimator)
            status_label = '<span>Fitted</span>'
            is_fitted_css_class = 'fitted'
        except NotFittedError:
            status_label = '<span>Not fitted</span>'
            is_fitted_css_class = ''
    is_fitted_icon = f'<span class="sk-estimator-doc-link {is_fitted_css_class}">i{status_label}</span>'
    with closing(StringIO()) as out:
        container_id = _CONTAINER_ID_COUNTER.get_id()
        style_template = Template(_CSS_STYLE)
        style_with_id = style_template.substitute(id=container_id)
        estimator_str = str(estimator)
        fallback_msg = 'In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.'
        html_template = f'<style>{style_with_id}</style><div id="{container_id}" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>{html.escape(estimator_str)}</pre><b>{fallback_msg}</b></div><div class="sk-container" hidden>'
        out.write(html_template)
        _write_estimator_html(out, estimator, estimator.__class__.__name__, estimator_str, first_call=True, is_fitted_css_class=is_fitted_css_class, is_fitted_icon=is_fitted_icon)
        out.write('</div></div>')
        html_output = out.getvalue()
        return html_output