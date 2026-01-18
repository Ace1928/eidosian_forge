import re
from email.errors import HeaderParseError
from email.parser import HeaderParser
from inspect import cleandoc
from django.urls import reverse
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
def create_reference_role(rolename, urlbase):
    is_case_sensitive = rolename in ['template', 'view']

    def _role(name, rawtext, text, lineno, inliner, options=None, content=None):
        if options is None:
            options = {}
        node = docutils.nodes.reference(rawtext, text, refuri=urlbase % (inliner.document.settings.link_base, text if is_case_sensitive else text.lower()), **options)
        return ([node], [])
    docutils.parsers.rst.roles.register_canonical_role(rolename, _role)