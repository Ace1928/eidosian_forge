import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
def gen_convo_ul(conversations):
    """
    Generate the ul section of the HTML for the conversations.

    :param conversation: The conversation to be rendered (after pre-processing)

    :return: The string generating the list in HTML
    """
    ul_str = f'\t<ul>\n'
    for speaker, speech in conversations:
        if speaker == END_OF_CONVO:
            ul_str += f'\n\t  <li class="breaker"><hr/></li>\n'
        else:
            ul_str += f'\n    <li>\n        <div class="{speaker}_img_div">\n            <img class="{speaker}_img">\n        </div>\n        <div class="{speaker}_p_div">\n            <p class="{speaker}">{speech}</p>\n        </div>\n        <div class="clear"></div>\n    </li>\n    '
    ul_str += '\t</ul>'
    return ul_str