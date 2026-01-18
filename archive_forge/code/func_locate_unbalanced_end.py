import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def locate_unbalanced_end(unbalanced_end, pre_delete, post_delete):
    """ like locate_unbalanced_start, except handling end tags and
    possibly moving the point earlier in the document.  """
    while 1:
        if not unbalanced_end:
            break
        finding = unbalanced_end[-1]
        finding_name = finding.split()[0].strip('<>/')
        if not pre_delete:
            break
        next = pre_delete[-1]
        if next is DEL_END or not next.startswith('</'):
            break
        name = next.split()[0].strip('<>/')
        if name == 'ins' or name == 'del':
            break
        if name == finding_name:
            unbalanced_end.pop()
            post_delete.insert(0, pre_delete.pop())
        else:
            break