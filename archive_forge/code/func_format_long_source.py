import re
import html
from paste.util import PySourceColor
def format_long_source(self, source, long_source):
    q_long_source = str2html(long_source, False, 4, True)
    q_source = str2html(source, True, 0, False)
    return '<code style="display: none" class="source" source-type="long"><a class="switch_source" onclick="return switch_source(this, \'long\')" href="#">&lt;&lt;&nbsp; </a>%s</code><code class="source" source-type="short"><a onclick="return switch_source(this, \'short\')" class="switch_source" href="#">&gt;&gt;&nbsp; </a>%s</code>' % (q_long_source, q_source)