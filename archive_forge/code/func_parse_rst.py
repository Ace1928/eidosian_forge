import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
def parse_rst(text, default_reference_context, thing_being_parsed=None):
    """
    Convert the string from reST to an XHTML fragment.
    """
    overrides = {'doctitle_xform': True, 'initial_header_level': 3, 'default_reference_context': default_reference_context, 'link_base': reverse('django-admindocs-docroot').rstrip('/'), 'raw_enabled': False, 'file_insertion_enabled': False}
    thing_being_parsed = thing_being_parsed and '<%s>' % thing_being_parsed
    source = '\n.. default-role:: cmsreference\n\n%s\n\n.. default-role::\n'
    parts = docutils.core.publish_parts(source % text, source_path=thing_being_parsed, destination_path=None, writer_name='html', settings_overrides=overrides)
    return mark_safe(parts['fragment'])