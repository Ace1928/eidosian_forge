import re
from django.core.exceptions import ValidationError
from django.forms.utils import RenderableFieldMixin, pretty_name
from django.forms.widgets import MultiWidget, Textarea, TextInput
from django.utils.functional import cached_property
from django.utils.html import format_html, html_safe
from django.utils.translation import gettext_lazy as _
def as_widget(self, widget=None, attrs=None, only_initial=False):
    """
        Render the field by rendering the passed widget, adding any HTML
        attributes passed as attrs. If a widget isn't specified, use the
        field's default widget.
        """
    widget = widget or self.field.widget
    if self.field.localize:
        widget.is_localized = True
    attrs = attrs or {}
    attrs = self.build_widget_attrs(attrs, widget)
    if self.auto_id and 'id' not in widget.attrs:
        attrs.setdefault('id', self.html_initial_id if only_initial else self.auto_id)
    if only_initial and self.html_initial_name in self.form.data:
        value = self.form._widget_data_value(self.field.hidden_widget(), self.html_initial_name)
    else:
        value = self.value()
    return widget.render(name=self.html_initial_name if only_initial else self.html_name, value=value, attrs=attrs, renderer=self.form.renderer)