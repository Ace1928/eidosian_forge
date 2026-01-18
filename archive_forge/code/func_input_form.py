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
def input_form(tbid, debug_info):
    return '\n<form action="#" method="POST"\n onsubmit="return submitInput($(\'submit_%(tbid)s\'), %(tbid)s)">\n<div id="exec-output-%(tbid)s" style="width: 95%%;\n padding: 5px; margin: 5px; border: 2px solid #000;\n display: none"></div>\n<input type="text" name="input" id="debug_input_%(tbid)s"\n style="width: 100%%"\n autocomplete="off" onkeypress="upArrow(this, event)"><br>\n<input type="submit" value="Execute" name="submitbutton"\n onclick="return submitInput(this, %(tbid)s)"\n id="submit_%(tbid)s"\n input-from="debug_input_%(tbid)s"\n output-to="exec-output-%(tbid)s">\n<input type="submit" value="Expand"\n onclick="return expandInput(this)">\n</form>\n ' % {'tbid': tbid}