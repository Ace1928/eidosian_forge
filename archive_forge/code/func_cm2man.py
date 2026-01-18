from docutils.core import publish_cmdline, default_description
from recommonmark.parser import CommonMarkParser
def cm2man():
    description = 'Generate a manpage from markdown sources. ' + default_description
    publish_cmdline(writer_name='manpage', parser=CommonMarkParser(), description=description)