import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
def render(self, info, format=None, fragment=False, template=None):
    """Render the template to a string using the provided info."""
    kwargs = self._get_render_options(format=format, fragment=fragment)
    return self.transform(info, template).render(**kwargs)