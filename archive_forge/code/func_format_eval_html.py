import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
def format_eval_html(exc_data, base_path, counter):
    short_formatter = EvalHTMLFormatter(base_path=base_path, counter=counter, include_reusable=False)
    short_er = short_formatter.format_collected_data(exc_data)
    long_formatter = EvalHTMLFormatter(base_path=base_path, counter=counter, show_hidden_frames=True, show_extra_data=False, include_reusable=False)
    long_er = long_formatter.format_collected_data(exc_data)
    text_er = formatter.format_text(exc_data, show_hidden_frames=True)
    if short_formatter.filter_frames(exc_data.frames) != long_formatter.filter_frames(exc_data.frames):
        full_traceback_html = '\n    <br>\n    <script type="text/javascript">\n    show_button(\'full_traceback\', \'full traceback\')\n    </script>\n    <div id="full_traceback" class="hidden-data">\n    %s\n    </div>\n        ' % long_er
    else:
        full_traceback_html = ''
    return '\n    %s\n    %s\n    <br>\n    <script type="text/javascript">\n    show_button(\'text_version\', \'text version\')\n    </script>\n    <div id="text_version" class="hidden-data">\n    <textarea style="width: 100%%" rows=10 cols=60>%s</textarea>\n    </div>\n    ' % (short_er, full_traceback_html, html.escape(text_er))