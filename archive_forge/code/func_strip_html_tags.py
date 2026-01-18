from .core import *
from .helpers import DelimitedList, any_open_tag, any_close_tag
from datetime import datetime
@staticmethod
def strip_html_tags(s: str, l: int, tokens: ParseResults):
    """Parse action to remove HTML tags from web page HTML source

        Example::

            # strip HTML links from normal text
            text = '<td>More info at the <a href="https://github.com/pyparsing/pyparsing/wiki">pyparsing</a> wiki page</td>'
            td, td_end = make_html_tags("TD")
            table_text = td + SkipTo(td_end).set_parse_action(pyparsing_common.strip_html_tags)("body") + td_end
            print(table_text.parse_string(text).body)

        Prints::

            More info at the pyparsing wiki page
        """
    return pyparsing_common._html_stripper.transform_string(tokens[0])