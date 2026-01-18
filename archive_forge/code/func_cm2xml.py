from docutils.core import publish_cmdline, default_description
from recommonmark.parser import CommonMarkParser
def cm2xml():
    description = 'Generate XML document from markdown sources. ' + default_description
    publish_cmdline(writer_name='xml', parser=CommonMarkParser(), description=description)