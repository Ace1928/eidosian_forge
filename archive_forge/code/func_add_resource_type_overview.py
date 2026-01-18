import inspect
import jmespath
from botocore.compat import six
def add_resource_type_overview(section, resource_type, description, intro_link=None):
    section.style.new_line()
    section.write('.. rst-class:: admonition-title')
    section.style.new_line()
    section.style.new_line()
    section.write(resource_type)
    section.style.new_line()
    section.style.new_line()
    section.write(description)
    section.style.new_line()
    if intro_link is not None:
        section.write('For more information about %s refer to the :ref:`Resources Introduction Guide<%s>`.' % (resource_type.lower(), intro_link))
        section.style.new_line()