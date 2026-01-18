from docutils.core import publish_cmdline, default_description
from recommonmark.parser import CommonMarkParser
def cm2html():
    description = 'Generate html document from markdown sources. ' + default_description
    publish_cmdline(writer_name='html', parser=CommonMarkParser(), description=description)