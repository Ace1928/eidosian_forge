import re
from django.core.exceptions import ValidationError
from django.forms.utils import RenderableFieldMixin, pretty_name
from django.forms.widgets import MultiWidget, Textarea, TextInput
from django.utils.functional import cached_property
from django.utils.html import format_html, html_safe
from django.utils.translation import gettext_lazy as _
@property
def auto_id(self):
    """
        Calculate and return the ID attribute for this BoundField, if the
        associated Form has specified auto_id. Return an empty string otherwise.
        """
    auto_id = self.form.auto_id
    if auto_id and '%s' in str(auto_id):
        return auto_id % self.html_name
    elif auto_id:
        return self.html_name
    return ''