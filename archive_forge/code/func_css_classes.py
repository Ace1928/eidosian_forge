import re
from django.core.exceptions import ValidationError
from django.forms.utils import RenderableFieldMixin, pretty_name
from django.forms.widgets import MultiWidget, Textarea, TextInput
from django.utils.functional import cached_property
from django.utils.html import format_html, html_safe
from django.utils.translation import gettext_lazy as _
def css_classes(self, extra_classes=None):
    """
        Return a string of space-separated CSS classes for this field.
        """
    if hasattr(extra_classes, 'split'):
        extra_classes = extra_classes.split()
    extra_classes = set(extra_classes or [])
    if self.errors and hasattr(self.form, 'error_css_class'):
        extra_classes.add(self.form.error_css_class)
    if self.field.required and hasattr(self.form, 'required_css_class'):
        extra_classes.add(self.form.required_css_class)
    return ' '.join(extra_classes)