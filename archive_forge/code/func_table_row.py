import pyparsing as pp
import urllib.request
from pprint import pprint
def table_row(start_tag, end_tag):
    body = start_tag.tag_body
    body.addParseAction(pp.tokenMap(str.strip), pp.tokenMap(strip_html))
    row = pp.Group(tr.suppress() + pp.ZeroOrMore(start_tag.suppress() + body + end_tag.suppress()) + tr_end.suppress())
    return row