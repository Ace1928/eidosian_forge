import re
from django.core.exceptions import ValidationError
from django.forms.utils import RenderableFieldMixin, pretty_name
from django.forms.widgets import MultiWidget, Textarea, TextInput
from django.utils.functional import cached_property
from django.utils.html import format_html, html_safe
from django.utils.translation import gettext_lazy as _
def as_hidden(self, attrs=None, **kwargs):
    """
        Return a string of HTML for representing this as an <input type="hidden">.
        """
    return self.as_widget(self.field.hidden_widget(), attrs, **kwargs)