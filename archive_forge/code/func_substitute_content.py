import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def substitute_content(content, vars, filename='<string>', use_cheetah=False, template_renderer=None):
    global Cheetah
    v = standard_vars.copy()
    v.update(vars)
    vars = v
    if template_renderer is not None:
        return template_renderer(content, vars, filename=filename)
    if not use_cheetah:
        tmpl = LaxTemplate(content)
        try:
            return tmpl.substitute(TypeMapper(v))
        except Exception as e:
            _add_except(e, ' in file %s' % filename)
            raise
    if Cheetah is None:
        import Cheetah.Template
    tmpl = Cheetah.Template.Template(source=content, searchList=[vars])
    return careful_sub(tmpl, vars, filename)