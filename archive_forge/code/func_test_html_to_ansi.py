import datetime
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import unittest
import warnings
from humanfriendly import (
from humanfriendly.case import CaseInsensitiveDict, CaseInsensitiveKey
from humanfriendly.cli import main
from humanfriendly.compat import StringIO
from humanfriendly.decorators import cached
from humanfriendly.deprecation import DeprecationProxy, define_aliases, deprecated_args, get_aliases
from humanfriendly.prompts import (
from humanfriendly.sphinx import (
from humanfriendly.tables import (
from humanfriendly.terminal import (
from humanfriendly.terminal.html import html_to_ansi
from humanfriendly.terminal.spinners import AutomaticSpinner, Spinner
from humanfriendly.testing import (
from humanfriendly.text import (
from humanfriendly.usage import (
from mock import MagicMock
def test_html_to_ansi(self):
    """Test the :func:`humanfriendly.terminal.html_to_ansi()` function."""
    assert html_to_ansi('Just some plain text') == 'Just some plain text'
    assert html_to_ansi('<a href="https://python.org">python.org</a>') == '\x1b[0m\x1b[4;94mpython.org\x1b[0m (\x1b[0m\x1b[4;94mhttps://python.org\x1b[0m)'
    assert html_to_ansi('<a href="mailto:peter@peterodding.com">peter@peterodding.com</a>') == '\x1b[0m\x1b[4;94mpeter@peterodding.com\x1b[0m'
    assert html_to_ansi("Let's try <b>bold</b>") == "Let's try \x1b[0m\x1b[1mbold\x1b[0m"
    assert html_to_ansi('Let\'s try <span style="font-weight: bold">bold</span>') == "Let's try \x1b[0m\x1b[1mbold\x1b[0m"
    assert html_to_ansi("Let's try <i>italic</i>") == "Let's try \x1b[0m\x1b[3mitalic\x1b[0m"
    assert html_to_ansi('Let\'s try <span style="font-style: italic">italic</span>') == "Let's try \x1b[0m\x1b[3mitalic\x1b[0m"
    assert html_to_ansi("Let's try <ins>underline</ins>") == "Let's try \x1b[0m\x1b[4munderline\x1b[0m"
    assert html_to_ansi('Let\'s try <span style="text-decoration: underline">underline</span>') == "Let's try \x1b[0m\x1b[4munderline\x1b[0m"
    assert html_to_ansi("Let's try <s>strike-through</s>") == "Let's try \x1b[0m\x1b[9mstrike-through\x1b[0m"
    assert html_to_ansi('Let\'s try <span style="text-decoration: line-through">strike-through</span>') == "Let's try \x1b[0m\x1b[9mstrike-through\x1b[0m"
    assert html_to_ansi("Let's try <code>pre-formatted</code>") == "Let's try \x1b[0m\x1b[33mpre-formatted\x1b[0m"
    assert html_to_ansi('Let\'s try <span style="color: #AABBCC">text colors</s>') == "Let's try \x1b[0m\x1b[38;2;170;187;204mtext colors\x1b[0m"
    assert html_to_ansi('Let\'s try <span style="background-color: rgb(50, 50, 50)">background colors</s>') == "Let's try \x1b[0m\x1b[48;2;50;50;50mbackground colors\x1b[0m"
    assert html_to_ansi("Let's try some<br>line<br>breaks") == "Let's try some\nline\nbreaks"
    assert html_to_ansi('&#38;') == '&'
    assert html_to_ansi('&amp;') == '&'
    assert html_to_ansi('&gt;') == '>'
    assert html_to_ansi('&lt;') == '<'
    assert html_to_ansi('&#x26;') == '&'

    def callback(text):
        return text.replace(':wink:', ';-)')
    assert ':wink:' not in html_to_ansi('<b>:wink:</b>', callback=callback)
    assert ':wink:' in html_to_ansi('<code>:wink:</code>', callback=callback)
    assert html_to_ansi(u'\n            Tweakers zit er idd nog steeds:<br><br>\n            peter@peter-work&gt; curl -s <a href="tweakers.net">tweakers.net</a> | grep -i hosting<br>\n            &lt;a href="<a href="http://www.true.nl/webhosting/">http://www.true.nl/webhosting/</a>"\n                rel="external" id="true" title="Hosting door True"&gt;&lt;/a&gt;<br>\n            Hosting door &lt;a href="<a href="http://www.true.nl/vps/">http://www.true.nl/vps/</a>"\n                title="VPS hosting" rel="external"&gt;True</a>\n        ')