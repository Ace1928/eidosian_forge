import warnings
from bleach import ALLOWED_ATTRIBUTES, ALLOWED_TAGS, clean
from traitlets import Any, Bool, List, Set, Unicode
from .base import Preprocessor
def sanitize_html_tags(self, html_str):
    """
        Sanitize a string containing raw HTML tags.
        """
    kwargs = {'tags': self.tags, 'attributes': self.attributes, 'strip': self.strip, 'strip_comments': self.strip_comments}
    if _USE_BLEACH_CSS_SANITIZER:
        css_sanitizer = CSSSanitizer(allowed_css_properties=self.styles)
        kwargs.update(css_sanitizer=css_sanitizer)
    elif _USE_BLEACH_STYLES:
        kwargs.update(styles=self.styles)
    return clean(html_str, **kwargs)